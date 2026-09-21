import asyncio
import json
import queue
import threading
import os
import re
import unicodedata
import websockets
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from io import BytesIO
import soundfile as sf
from difflib import SequenceMatcher
from scipy.spatial.distance import euclidean
from resemblyzer import VoiceEncoder, preprocess_wav

SERVER_URI = "ws://localhost:8765"

# ==============================================================================
# CONFIGURACIÓN DE AUDIO Y WAKE WORDS
# ==============================================================================
SAMPLE_RATE = 16000
CHUNK_DURATION = 4  # Ventana de captura continua en segundos
WAKE_WORDS_BASE = ["alex", "alexa", "robin", "ales", "robot alex", "robot robin"]

def normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas, elimina tildes/acentos y quita puntuación."""
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

WAKE_WORDS_CLEAN = [normalizar_texto(w) for w in WAKE_WORDS_BASE]

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

ROBOT_RECENT_TEXTS = []
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
            dist = euclidean(firma_actual, firma_conocida)
            if dist < distancia_min:
                distancia_min = dist
                mejor_match = nombre
                
        if distancia_min > 0.85:
            return "Desconocido"
            
        return mejor_match

    except Exception as e:
        print(f"⚠️ Error identificando voz con Resemblyzer: {e}")
        return "Desconocido"

def contiene_palabra_despertar(text: str) -> bool:
    """Verifica si el texto normalizado contiene alguna de las wake words."""
    text_clean = normalizar_texto(text)
    return any(
        text_clean.startswith(ww) or f" {ww} " in f" {text_clean} "
        for ww in WAKE_WORDS_CLEAN
    )

def is_similar_to_robot_speech(text: str, threshold: float = 0.45) -> bool:
    """Filtro Anti-Eco para ignorar lo que el robot acaba de decir."""
    text_clean = normalizar_texto(text)
    for robot_phrase in ROBOT_RECENT_TEXTS:
        rf_clean = normalizar_texto(robot_phrase)
        similarity = SequenceMatcher(None, text_clean, rf_clean).ratio()
        if similarity >= threshold or (len(text_clean) > 6 and text_clean in rf_clean) or (len(rf_clean) > 6 and rf_clean in text_clean):
            return True
    return False

# ==============================================================================
# 3. TAREAS ASÍNCRONAS Y CAPTURA (USANDO SOUNDDEVICE)
# ==============================================================================
async def listen_robot_broadcaster():
    """Sincronización anti-eco mediante canal WebSocket."""
    while True:
        try:
            async with websockets.connect(SERVER_URI) as ws:
                await ws.send(json.dumps({"type": "REGISTER_CAPTURER"}))
                print("🔗 [CAPTURADOR UNIFICADO] Conectado al servidor con Anti-Eco y Wake Word.")
                
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "ROBOT_SPOKE":
                        phrase = data.get("text", "")
                        if phrase:
                            ROBOT_RECENT_TEXTS.append(phrase)
                            if len(ROBOT_RECENT_TEXTS) > 8:
                                ROBOT_RECENT_TEXTS.pop(0)
        except Exception:
            await asyncio.sleep(2)

def background_mic_worker_sounddevice():
    """Captura continua desde el micrófono usando sounddevice en lugar de PyAudio."""
    print("[MIC] Captura de micrófono activa (sounddevice)...")
    while True:
        try:
            # Graba en bloques continuos
            recording = sd.rec(
                int(CHUNK_DURATION * SAMPLE_RATE), 
                samplerate=SAMPLE_RATE, 
                channels=1, 
                dtype='int16'
            )
            sd.wait()
            
            # Convierte el buffer de numpy a un objeto AudioData de SpeechRecognition
            audio_bytes = recording.tobytes()
            audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
            AUDIO_QUEUE.put(audio_data)
            
        except Exception as e:
            print(f"⚠️ Error en captura de micrófono: {e}")

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

                    # Filtro 2: Anti-eco (Descomentar si se requiere)
                    # if is_similar_to_robot_speech(clean_text):
                    #    print(f"🛑 [ANTI-ECO] Audio del robot ignorado: \"{clean_text[:35]}...\"")
                    #    continue

                    # Filtro 3: Comprobar la Palabra de Despertar (normalizada)
                    if not contiene_palabra_despertar(clean_text):
                        print(f"💤 [IGNORADO - Sin Wake Word]: \"{clean_text}\"")
                        continue

                    # 2. Identificar a la persona usando Resemblyzer
                    speaker_name = await asyncio.to_thread(
                        reconocer_hablante_desde_audio_data, audio
                    )

                    label_log = speaker_name if speaker_name != "Desconocido" else "Persona No Registrada"
                    print(f"\n🔔 [WAKE WORD DETECTADA] 🎤 [{label_log}]: \"{clean_text}\"")
                    
                    # 3. Preparar mensaje para el servidor
                    payload = {
                        "type": "HUMAN_INPUT",
                        "speaker": speaker_name if speaker_name != "Desconocido" else "",
                        "is_known_speaker": speaker_name != "Desconocido",
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
    print(f"[MIC] ¡Sistema listo! Esperando palabras de activación: {WAKE_WORDS_BASE}...\n")

    # Iniciar el hilo de grabación con sounddevice
    threading.Thread(target=background_mic_worker_sounddevice, daemon=True).start()

    await asyncio.gather(
        process_audio_queue(),
        listen_robot_broadcaster()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CAPTURADOR] Detenido.")