import asyncio
import json
import os
import re
import time
import unicodedata
from io import BytesIO
from difflib import SequenceMatcher 
import numpy as np
import scipy.spatial.distance
import soundfile as sf
import websockets
import speech_recognition as sr
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav

load_dotenv()
SERVER_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8765")

ROBOT_RECENT_TEXTS = []
WAKE_WORDS_MAP = {
    "robots": ["robots", "robot"],
    "lumi": ["lumi", "lumy", "robot lumi"],
    "nova": ["nova", "novy", "robot nova", "no va"]
}

def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

def es_eco_del_robot(texto_capturado: str) -> bool:
    clean_cap = normalizar_texto(texto_capturado)
    if not clean_cap:
        return False
    for frase_robot in list(ROBOT_RECENT_TEXTS):
        clean_robot = normalizar_texto(frase_robot)
        ratio = SequenceMatcher(None, clean_cap, clean_robot).ratio()
        if ratio > 0.60 or clean_cap in clean_robot:
            return True
    return False

# ==============================================================================
# BASE DE DATOS DE VOCES
# ==============================================================================
print("Cargando modelo de huellas de voz (Resemblyzer)...")
encoder = VoiceEncoder()

def extraer_firma_desde_path(file_path):
    try:
        wav = preprocess_wav(file_path)
        return encoder.embed_utterance(wav)
    except Exception as e:
        print(f"⚠️ Error cargando {file_path}: {e}")
        return None

def cargar_voces_conocidas():
    voces_config = {
        "Paula": ["voices/paula_ref2.wav", "voices/paula_ref3.wav", "voices/paula_ref4.wav"],
        "Loreto": ["voices/loreto_ref.wav", "voices/loreto_ref2.wav", "voices/loreto_ref3.wav", "voices/loreto_ref4.wav"],
        "Liany": ["voices/liany_ref.wav", "voices/liany_ref2.wav", "voices/liany_ref3.wav", "voices/liany_ref4.wav"],
        "Juan Jesus": ["voices/juanje_ref.wav", "voices/juanje_ref2.wav", "voices/juanje_ref3.wav", "voices/juanje_ref4.wav"],
        "Eva": ["voices/eva_ref.wav", "voices/eva_ref2.wav", "voices/eva_ref3.wav", "voices/eva_ref4.wav"]
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
    return voces_db

VOCES_CONOCIDAS = cargar_voces_conocidas()

def reconocer_hablante_desde_audio_data(audio_data: sr.AudioData) -> tuple[str, float]:
    try:
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        wav, sr_rate = sf.read(BytesIO(wav_bytes))
        wav_preprocessed = preprocess_wav(wav, source_sr=sr_rate)
        firma_actual = encoder.embed_utterance(wav_preprocessed)
        
        mejor_match = "Desconocido"
        max_similitud = 0.0
        UMBRAL_CERTEZA = 0.75 

        for nombre, firma_conocida in VOCES_CONOCIDAS.items():
            dot_product = np.dot(firma_actual, firma_conocida)
            norm_a = np.linalg.norm(firma_actual)
            norm_b = np.linalg.norm(firma_conocida)
            
            if norm_a > 0 and norm_b > 0:
                similitud = dot_product / (norm_a * norm_b)
                if similitud > max_similitud:
                    max_similitud = similitud
                    mejor_match = nombre
        
        porcentaje = float(max_similitud * 100)
        if max_similitud < UMBRAL_CERTEZA:
            return "Desconocido", porcentaje

        return mejor_match, porcentaje
    except Exception as e:
        return "Desconocido", 0.0

def extraer_target_wake_word(text: str) -> str:
    text_clean = normalizar_texto(text)
    for target_robot, aliases in WAKE_WORDS_MAP.items():
        for alias in aliases:
            alias_clean = normalizar_texto(alias)
            if text_clean == alias_clean or f" {alias_clean} " in f" {text_clean} " or text_clean.startswith(alias_clean + " "):
                return target_robot
    return None

async def enviar_evento_servidor(payload: dict):
    try:
        async with websockets.connect(SERVER_URI) as ws:
            await ws.send(json.dumps(payload))
    except Exception as e:
        print(f"⚠️ Error enviando evento: {e}")

async def listen_server_broadcasts():
    while True:
        try:
            async with websockets.connect(SERVER_URI) as ws:
                await ws.send(json.dumps({"type": "REGISTER_CAPTURER"}))
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") == "ROBOT_SPOKE":
                        texto = data.get("text", "")
                        if texto:
                            ROBOT_RECENT_TEXTS.append(texto)
                            if len(ROBOT_RECENT_TEXTS) > 5:
                                ROBOT_RECENT_TEXTS.pop(0)
        except Exception:
            await asyncio.sleep(2)

# ==============================================================================
# FLUJO PASO A PASO
# ==============================================================================
async def flujo_escucha_principal():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    mic = sr.Microphone()

    with mic as source:
        print("🎛️ Calibrando micro...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("\n[PASO 1] 👂 Esperando palabra clave (LUMI / NOVA)...")

    while True:
        try:
            # 1. ESCUCHA DE WAKE WORD (Corte rápido de pausa)
            recognizer.pause_threshold = 0.6
            with mic as source:
                audio_ww = await asyncio.to_thread(
                    recognizer.listen, source, timeout=None, phrase_time_limit=3
                )

            raw_ww = await asyncio.to_thread(
                recognizer.recognize_google, audio_ww, language="es-ES"
            )
            clean_ww = raw_ww.strip()

            if not clean_ww or es_eco_del_robot(clean_ww):
                continue

            target_robot = extraer_target_wake_word(clean_ww)

            if target_robot:
                print(f"\n✅ Wake Word detectada: \"{clean_ww}\" -> Target: {target_robot.upper()}")

                # 2. ENVIAR MENSAJE PARA PANTALLA LISTENING AL INSTANTE
                print(f"⚡ [PASO 2] Enviando HUMAN_LISTENING_START para {target_robot.upper()}...")
                await enviar_evento_servidor({
                    "type": "HUMAN_LISTENING_START",
                    "target": target_robot
                })

                # 3. ACTIVAR ESCUCHA DE LA ORDEN/PREGUNTA
                print(f"🎤 [PASO 3] Robot en pantalla LISTENING. Escuchando la pregunta...")
                recognizer.pause_threshold = 1.8  # Tiempo cómodo para decir la pregunta

                try:
                    with mic as source:
                        audio_pregunta = await asyncio.to_thread(
                            recognizer.listen, source, timeout=7.0, phrase_time_limit=10
                        )

                    raw_pregunta = await asyncio.to_thread(
                        recognizer.recognize_google, audio_pregunta, language="es-ES"
                    )
                    clean_pregunta = raw_pregunta.strip()

                    if clean_pregunta and not es_eco_del_robot(clean_pregunta):
                        # Identificar hablante
                        speaker_name, certeza = await asyncio.to_thread(
                            reconocer_hablante_desde_audio_data, audio_pregunta
                        )
                        es_conocido = speaker_name != "Desconocido"

                        # 4. ENVIAR HUMAN_INPUT (Pasa a PROCESSING)
                        print(f"🚀 [PASO 4] Enviando respuesta a servidor (Entrando en PROCESSING)...")
                        print(f"   💬 Texto: \"{clean_pregunta}\" | Speaker: {speaker_name}")

                        payload = {
                            "type": "HUMAN_INPUT",
                            "speaker": speaker_name if es_conocido else "Persona",
                            "is_known_speaker": es_conocido,
                            "confidence": round(float(certeza), 2),
                            "target": target_robot,
                            "text": clean_pregunta
                        }
                        await enviar_evento_servidor(payload)
                    else:
                        print("⚠️ No se entendió la pregunta. Volviendo a esperar Wake Word...")

                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    print("⏱️ Tiempo de espera agotado sin pregunta. Volviendo a estado reposo...")

                print("\n[PASO 1] 👂 Esperando palabra clave (LUMI / NOVA)...")

        except sr.UnknownValueError:
            pass
        except Exception as e:
            await asyncio.sleep(0.1)

async def main():
    await asyncio.gather(
        listen_server_broadcasts(),
        flujo_escucha_principal()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CAPTURADOR] Detenido.")