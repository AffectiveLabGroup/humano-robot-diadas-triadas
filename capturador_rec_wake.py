import asyncio
import json
import os
import queue
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

# Cargar variables de entorno desde el archivo .env
load_dotenv()

SERVER_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8765")

# ==============================================================================
# CONFIGURACIÓN DE AUDIO, WAKE WORDS Y VENTANA DE ATENCIÓN
# ==============================================================================
ROBOT_RECENT_TEXTS = []

WAKE_WORDS_MAP = {
    "robots": ["robots", "robot"],
    "alex": ["alex", "alexa", "ales", "robot alex"],
    "robin": ["robin", "rovin", "robot robin"]
}

ATTENTION_WINDOW_SECONDS = 6.0  # Segundos que el robot se queda escuchando tras decir su nombre
ACTIVE_TARGET = None            # Robot que está actualmente en estado "atento" ('alex' o 'robin')
LAST_WAKE_WORD_TIME = 0.0       # Marca de tiempo (timestamp) de cuando se dijo la wake word

def normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas, elimina tildes/acentos y quita puntuación."""
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

def es_eco_del_robot(texto_capturado: str) -> bool:
    """Compara si la frase escuchada se parece a lo que acaba de decir un robot."""
    clean_cap = normalizar_texto(texto_capturado)
    if not clean_cap:
        return False

    for frase_robot in list(ROBOT_RECENT_TEXTS):
        clean_robot = normalizar_texto(frase_robot)
        
        # Coincidencia por similitud de subsecuencia (> 60% igual = Eco)
        ratio = SequenceMatcher(None, clean_cap, clean_robot).ratio()
        if ratio > 0.60 or clean_cap in clean_robot:
            return True
    return False

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
        else:
            print(f"   ⚠️ No se encontraron muestras válidas para: {nombre}")
            
    return voces_db

VOCES_CONOCIDAS = cargar_voces_conocidas()
AUDIO_QUEUE = queue.Queue()

# ==============================================================================
# 2. FUNCIONES DE RECONOCIMIENTO Y FILTROS
# ==============================================================================
def reconocer_hablante_desde_audio_data(audio_data: sr.AudioData) -> tuple[str, float]:
    """
    Extrae la huella del objeto AudioData y calcula la similitud del coseno (0 a 100%).
    Retorna una tupla: (Nombre_o_Desconocido, Porcentaje_Similitud).
    """
    try:
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        wav, sr_rate = sf.read(BytesIO(wav_bytes))
        
        wav_preprocessed = preprocess_wav(wav, source_sr=sr_rate)
        firma_actual = encoder.embed_utterance(wav_preprocessed)
        
        mejor_match = "Desconocido"
        max_similitud = 0.0
        
        # Umbral mínimo de certeza (75% de similitud)
        UMBRAL_CERTEZA = 0.75 

        for nombre, firma_conocida in VOCES_CONOCIDAS.items():
            # Similitud del Coseno: 1.0 es idéntico, 0.0 es totalmente distinto
            dot_product = np.dot(firma_actual, firma_conocida)
            norm_a = np.linalg.norm(firma_actual)
            norm_b = np.linalg.norm(firma_conocida)
            
            if norm_a > 0 and norm_b > 0:
                similitud = dot_product / (norm_a * norm_b)
                
                if similitud > max_similitud:
                    max_similitud = similitud
                    mejor_match = nombre
        
        # Casteo explícito a float nativo de Python para prevenir errores de JSON
        porcentaje = float(max_similitud * 100)

        if max_similitud < UMBRAL_CERTEZA:
            return "Desconocido", porcentaje

        return mejor_match, porcentaje

    except Exception as e:
        print(f"⚠️ Error identificando voz con Resemblyzer: {e}")
        return "Desconocido", 0.0

def extraer_target_wake_word(text: str) -> str:
    """Identifica si la frase contiene una wake word y retorna 'alex', 'robin' o 'robots'."""
    text_clean = normalizar_texto(text)
    
    for target_robot, aliases in WAKE_WORDS_MAP.items():
        for alias in aliases:
            alias_clean = normalizar_texto(alias)
            if text_clean == alias_clean or f" {alias_clean} " in f" {text_clean} " or text_clean.startswith(alias_clean + " "):
                return target_robot
    return None

def es_solo_wake_word(text: str, target_robot: str) -> bool:
    """Devuelve True si el usuario solo pronunció el nombre del robot y nada más."""
    text_clean = normalizar_texto(text)
    aliases = WAKE_WORDS_MAP.get(target_robot, [])
    return any(text_clean == normalizar_texto(alias) for alias in aliases)

# ==============================================================================
# 3. ESCUCHA DE FRASES DEL ROBOT (Servidor -> Capturador)
# ==============================================================================
async def listen_server_broadcasts():
    """Conecta por WebSocket al orquestador para saber qué frases dicen los robots en tiempo real."""
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
# 4. CAPTURA Y PROCESAMIENTO CON VENTANA DE ATENCIÓN
# ==============================================================================
def audio_callback(recognizer, audio):
    """Callback invocado por SpeechRecognition cuando detecta voz."""
    AUDIO_QUEUE.put(audio)

async def process_audio_queue():
    """Procesa el audio, gestiona la ventana de atención y envía la señal al orquestador."""
    global ACTIVE_TARGET, LAST_WAKE_WORD_TIME
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

                    # FILTRO ANTI-ECO
                    if es_eco_del_robot(clean_text):
                        print(f" 🤫 [FILTRO ECO ROBOT - IGNORADO]: \"{clean_text}\"")
                        continue

                    now = time.time()
                    target_robot = extraer_target_wake_word(clean_text)

                    # --- LÓGICA DE CONTROL DE LA VENTANA DE ATENCIÓN ---
                    if target_robot:
                        ACTIVE_TARGET = target_robot
                        LAST_WAKE_WORD_TIME = now

                        if es_solo_wake_word(clean_text, target_robot):
                            print(f"\n👀 [WAKE WORD: {target_robot.upper()}]: Detectado nombre suelto. Escuchando durante {ATTENTION_WINDOW_SECONDS}s...")
                            continue

                    elif ACTIVE_TARGET and (now - LAST_WAKE_WORD_TIME < ATTENTION_WINDOW_SECONDS):
                        target_robot = ACTIVE_TARGET
                        print(f"⏱️ [VENTANA ACTIVA -> {target_robot.upper()}]: Mensaje capturado dentro del tiempo de espera.")

                    else:
                        print(f"💤 [IGNORADO - Sin Wake Word / Fuera de ventana]: \"{clean_text}\"")
                        continue

                    # Si llegamos aquí, la frase es válida. Consumimos la atención.
                    ACTIVE_TARGET = None 

                    # 2. Identificar a la persona usando la función con Similitud del Coseno
                    speaker_name, certeza = await asyncio.to_thread(
                        reconocer_hablante_desde_audio_data, audio
                    )

                    es_conocido = speaker_name != "Desconocido"
                    
                    # Log claro para depurar en consola
                    if es_conocido:
                        print(f"\n🔔 [ENVIANDO -> {target_robot.upper()}] 🎤 Hablante: {speaker_name} ({certeza:.1f}% certeza) | Frase: \"{clean_text}\"")
                    else:
                        print(f"\n🔔 [ENVIANDO -> {target_robot.upper()}] 🎤 Hablante: Persona No Registrada (Máx. Similitud: {certeza:.1f}%) | Frase: \"{clean_text}\"")
                    
                    # 3. Preparar paquete para el orquestador
                    payload = {
                        "type": "HUMAN_INPUT",
                        "speaker": speaker_name if es_conocido else "Persona",
                        "is_known_speaker": es_conocido,
                        "confidence": round(float(certeza), 2), # Aseguramos casteo a float nativo
                        "target": target_robot,
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
    print(f"[MIC] ¡Sistema listo! Esperando Wake Words con atención de {ATTENTION_WINDOW_SECONDS}s...\n")

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    
    recognizer.pause_threshold = 0.8 
    
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    stop_listening = recognizer.listen_in_background(mic, audio_callback)

    try:
        await asyncio.gather(
            listen_server_broadcasts(),
            process_audio_queue()
        )
    finally:
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CAPTURADOR] Detenido.")