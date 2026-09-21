import asyncio
import json
import queue
import threading
import os
import websockets
import speech_recognition as sr
import numpy as np
from io import BytesIO
import soundfile as sf
from difflib import SequenceMatcher
from scipy.spatial.distance import euclidean
from resemblyzer import VoiceEncoder, preprocess_wav

SERVER_URI = "ws://localhost:8765"

# ==============================================================================
# 1. INICIALIZACIÓN DE ENCODER Y BASE DE DATOS DE VOCES
# ==============================================================================
print("⏳ Cargando modelo de huellas de voz (Resemblyzer)...")
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
            print(f"  ✅ Voz cargada: {nombre} ({len(firmas)} muestras)")
        else:
            print(f"  ⚠️ No se encontraron muestras válidas para: {nombre}")
            
    return voces_db

VOCES_CONOCIDAS = cargar_voces_conocidas()

# Memoria local de frases del robot para el filtro anti-eco
ROBOT_RECENT_TEXTS = []
AUDIO_QUEUE = queue.Queue()

# ==============================================================================
# 2. FUNCIONES DE RECONOCIMIENTO Y FILTROS
# ==============================================================================
def reconocer_hablante_desde_audio_data(audio_data: sr.AudioData) -> str:
    """Extrae la huella del objeto AudioData en memoria y busca el match más cercano."""
    try:
        # Convertir AudioData a array de numpy normalizado para Resemblyzer
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        wav, sr_rate = sf.read(BytesIO(wav_bytes))
        
        # Preprocesar en memoria sin escribir a disco
        wav_preprocessed = preprocess_wav(wav, source_sr=sr_rate)
        firma_actual = encoder.embed_utterance(wav_preprocessed)
        
        mejor_match = "Desconocido"
        distancia_min = float("inf")
        
        # Comparar con la base de datos de firmas conocidas
        for nombre, firma_conocida in VOCES_CONOCIDAS.items():
            dist = euclidean(firma_actual, firma_conocida)
            if dist < distancia_min:
                distancia_min = dist
                mejor_match = nombre
                
        # Umbral de tolerancia (si la distancia es > 0.85, se considera hablante no registrado)
        if distancia_min > 0.85:
            return "H1"
            
        return mejor_match

    except Exception as e:
        print(f"⚠️ Error identificando voz con Resemblyzer: {e}")
        return "H1"

def is_similar_to_robot_speech(text: str, threshold: float = 0.45) -> bool:
    """Filtro Anti-Eco para ignorar lo que el robot acaba de decir."""
    text_clean = text.lower().strip()
    for robot_phrase in ROBOT_RECENT_TEXTS:
        rf_clean = robot_phrase.lower().strip()
        similarity = SequenceMatcher(None, text_clean, rf_clean).ratio()
        if similarity >= threshold or (len(text_clean) > 6 and text_clean in rf_clean) or (len(rf_clean) > 6 and rf_clean in text_clean):
            return True
    return False

# ==============================================================================
# 3. TAREAS ASÍNCRONAS Y CAPTURA
# ==============================================================================
async def listen_robot_broadcaster():
    """Sincronización anti-eco mediante canal WebSocket."""
    while True:
        try:
            async with websockets.connect(SERVER_URI) as ws:
                await ws.send(json.dumps({"type": "REGISTER_CAPTURER"}))
                print("🔗 [CAPTURADOR UNIFICADO] Conectado al servidor con Anti-Eco y Resemblyzer.")
                
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

def background_mic_worker(recognizer, mic):
    """Captura continua desde el micrófono ambiental."""
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                if audio:
                    AUDIO_QUEUE.put(audio)
        except Exception:
            pass

async def process_audio_queue():
    """Procesa el audio, transcribe el texto e identifica al hablante en paralelo."""
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
                    # Filtro 1: Ruido corto
                    words = clean_text.split()
                    if len(words) < 2 and len(clean_text) < 5:
                        continue

                    # Filtro 2: Anti-eco
                    if is_similar_to_robot_speech(clean_text):
                        print(f"🛑 [ANTI-ECO] Audio del robot ignorado: \"{clean_text[:35]}...\"")
                        continue

                    # 2. Identificar a la persona usando Resemblyzer en memoria
                    speaker_name = await asyncio.to_thread(
                        reconocer_hablante_desde_audio_data, audio
                    )

                    print(f"\n👉 🎤 [{speaker_name}]: \"{clean_text}\"")
                    
                    # 3. Enviar al Servidor Orquestador
                    async with websockets.connect(SERVER_URI) as ws:
                        payload = {
                            "type": "HUMAN_INPUT",
                            "speaker": speaker_name,
                            "text": clean_text
                        }
                        await ws.send(json.dumps(payload))
                        print(f"  ⚡ [{speaker_name}] -> Enviado al orquestador.")

            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"[!] Error procesando audio: {e}")
        
        await asyncio.sleep(0.05)

async def main():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.4
    recognizer.non_speaking_duration = 0.6
    recognizer.energy_threshold = 450
    recognizer.dynamic_energy_threshold = True

    mic = sr.Microphone()

    print("\n[MIC] Calibrando sonido ambiental (1 segundo)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("[MIC] ¡Sistema listo! Escuchando con reconocimiento de nombres...\n")

    threading.Thread(target=background_mic_worker, args=(recognizer, mic), daemon=True).start()

    await asyncio.gather(
        process_audio_queue(),
        listen_robot_broadcaster()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CAPTURADOR] Detenido.")