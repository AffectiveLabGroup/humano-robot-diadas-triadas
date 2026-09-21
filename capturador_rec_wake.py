import asyncio
import json
import os
import queue
import re
import unicodedata
from io import BytesIO
import numpy as np
import scipy.spatial.distance
import soundfile as sf
import websockets
import speech_recognition as sr
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav

# Cargar variables de entorno desde el archivo .env
load_dotenv()

SERVER_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8765")

# ==============================================================================
# CONFIGURACIÓN DE AUDIO Y WAKE WORDS MAPEADAS POR ROBOT
# ==============================================================================
WAKE_WORDS_MAP = {
    "alex": ["alex", "alexa", "ales", "robot alex"],
    "robin": ["robin", "rovin", "robot robin"]
}

def normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas, elimina tildes/acentos y quita puntuación."""
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

# ==============================================================================
# 1. INICIALIZACIÓN DE ENCODER Y BASE DE DATOS DE VOCES
# ==============================================================================
print("Cargando modelo de huellas de voz (Resemblyzer)...")
encoder = VoiceEncoder()

def extraer_firma_desde_path(file_path):
    """Extrae el vector de características de un archivo en disco."""
    try:
        wav = preprocess_wav(file_path)
        return encoder.embed_utterance(wav)
    except Exception as e:
        print(f"⚠️ Error cargando {file_path}: {e}")
        return None

def cargar_voces_conocidas():
    """Precarga las firmas promedio de cada persona al arrancar."""
    voces_config = {
        "Paula": ["voices/paula_ref.wav", "voices/paula_ref2.wav", "voices/paula_ref3.wav", "voices/paula_ref4.wav"],
        "Loreto": ["voices/loreto_ref.wav", "voices/loreto_ref2.wav", "voices/loreto_ref3.wav", "voices/loreto_ref4.wav"],
        "Liany": ["voices/liany_ref.wav", "voices/liany_ref2.wav", "voices/liany_ref3.wav", "voices/liany_ref4.wav"],
        "Juan Jesus": ["voices/juanje_ref.wav", "voices/juanje_ref2.wav", "voices/juanje_ref3.wav", "voices/juanje_ref4.wav"]
    }
    
    voces_db = {}
    for nombre, rutas in voces_config.items():
        firmas = []
        for r in rutas:
            if os.path.exists(r):
                emb = extraer_firma_desde_path(r)
                if emb is not None:
                    firmas.append(emb)
        if firmas:
            voces_db[nombre] = np.mean(firmas, axis=0)
            print(f"   ✅ Voz cargada: {nombre} ({len(firmas)} muestras)")
        else:
            print(f"   ⚠️ No se encontraron muestras válidas para: {nombre}")
            
    return voces_db

VOCES_CONOCIDAS = cargar_voces_conocidas()
AUDIO_QUEUE = queue.Queue()

# ==============================================================================
# 2. FUNCIONES DE RECONOCIMIENTO Y FILTROS
# ==============================================================================
def reconocer_hablante_desde_audio_data(audio_data: sr.AudioData) -> str:
    """Extrae la huella del objeto AudioData y busca el match más cercano."""
    try:
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        wav, sr_rate = sf.read(BytesIO(wav_bytes))
        
        wav_preprocessed = preprocess_wav(wav, source_sr=sr_rate)
        firma_actual = encoder.embed_utterance(wav_preprocessed)
        
        mejor_match = None
        distancia_min = float("inf")
        
        for nombre, firma_conocida in VOCES_CONOCIDAS.items():
            dist = scipy.spatial.distance.euclidean(firma_actual, firma_conocida)
            if dist < distancia_min:
                distancia_min = dist
                mejor_match = nombre
                
        if distancia_min > 0.85:
            return "Desconocido"
            
        return mejor_match

    except Exception as e:
        print(f"⚠️ Error identificando voz con Resemblyzer: {e}")
        return "Desconocido"

def extraer_target_wake_word(text: str) -> str:
    """
    Identifica si la frase contiene una wake word y retorna 'alex' o 'robin'.
    Retorna None si no hay coincidencia.
    """
    text_clean = normalizar_texto(text)
    
    for target_robot, aliases in WAKE_WORDS_MAP.items():
        for alias in aliases:
            alias_clean = normalizar_texto(alias)
            if alias_clean.startswith(text_clean) or f" {alias_clean} " in f" {text_clean} " or text_clean.startswith(alias_clean):
                return target_robot
    return None

# ==============================================================================
# 3. CAPTURA Y PROCESAMIENTO (PyAudio)
# ==============================================================================
def audio_callback(recognizer, audio):
    """Callback invocado por SpeechRecognition cuando detecta voz."""
    AUDIO_QUEUE.put(audio)

async def process_audio_queue():
    """Procesa el audio, valida la Wake Word y reconoce al hablante."""
    recognizer = sr.Recognizer()
    
    while True:
        if not AUDIO_QUEUE.empty():
            audio = AUDIO_QUEUE.get()
            
            try:
                # 1. Transcribir texto con Google Speech API
                raw_text = await asyncio.to_thread(
                    recognizer.recognize_google, audio, language="es-ES"
                )
                
                clean_text = raw_text.strip()
                if clean_text:
                    # Filtro 1: Ruido o frases extremadamente cortas
                    words = clean_text.split()
                    if len(words) < 2 and len(clean_text) < 5:
                        continue

                    # Filtro 2: Extraer el robot objetivo desde la Wake Word
                    target_robot = extraer_target_wake_word(clean_text)
                    if not target_robot:
                        print(f"💤 [IGNORADO - Sin Wake Word]: \"{clean_text}\"")
                        continue

                    # 2. Identificar a la persona usando Resemblyzer
                    speaker_name = await asyncio.to_thread(
                        reconocer_hablante_desde_audio_data, audio
                    )

                    label_log = speaker_name if speaker_name != "Desconocido" else "Persona No Registrada"
                    print(f"\n🔔 [WAKE WORD -> {target_robot.upper()}] 🎤 [{label_log}]: \"{clean_text}\"")
                    
                    # 3. Preparar paquete enviando el destinatario exacto
                    payload = {
                        "type": "HUMAN_INPUT",
                        "speaker": speaker_name if speaker_name != "Desconocido" else "",
                        "is_known_speaker": speaker_name != "Desconocido",
                        "target": target_robot,  # Envía 'alex' o 'robin'
                        "text": clean_text
                    }

                    # 4. Enviar al Servidor Orquestador
                    try:
                        async with websockets.connect(SERVER_URI) as ws:
                            await ws.send(json.dumps(payload))
                            print("   ⚡ Enviado al orquestador.")
                    except Exception as ws_err:
                        print(f"⚠️ Error al enviar payload por WebSocket: {ws_err}")

            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"[!] Error procesando audio: {e}")
        
        await asyncio.sleep(0.05)

async def main():
    print(f"[MIC] ¡Sistema listo! Esperando palabras de activación (Alex / Robin)...\n")

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    stop_listening = recognizer.listen_in_background(mic, audio_callback)

    try:
        await process_audio_queue()
    finally:
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CAPTURADOR] Detenido.")