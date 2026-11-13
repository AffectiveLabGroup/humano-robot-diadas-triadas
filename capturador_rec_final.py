"""
Capturador de audio con STT LOCAL en streaming (Vosk) + Whisper opcional.

Idea clave: Vosk va emitiendo resultados PARCIALES mientras el usuario habla.
En cuanto aparece "dani" / "gabi" en un parcial se envía HUMAN_LISTENING_START,
sin esperar a que termine la frase ni hacer ninguna petición de red.

Flujo:
  - "dani, yo opino que..." (>= 3 palabras tras el nombre) -> HUMAN_INPUT directo.
  - Solo "dani" -> pantalla de escucha y se espera la pregunta (6 s).
  - Si el parcial era un falso positivo -> HUMAN_LISTENING_TIMEOUT.

Variables de entorno (.env):
  WEBSOCKET_URI=ws://localhost:8765
  VOSK_MODEL_PATH=models/vosk-model-small-es-0.42
  VOSK_FIN_SILENCIO=1.2     # segundos de silencio que cierran la frase (más = admite pausas largas)
  VOSK_MAX_FRASE=30         # duración máxima de una frase en segundos
  MIC_INDEX=                # opcional: índice del micro (sounddevice)
  USE_WHISPER_FINAL=0       # 1 = refinar el texto final con faster-whisper
  WHISPER_SIZE=small        # tiny / base / small
  DEBUG_STT=0               # 1 = imprime los parciales de Vosk
"""
import asyncio
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav
from vosk import KaldiRecognizer, Model, SetLogLevel

try:  # solo existe en versiones recientes de Vosk
    from vosk import EndpointerMode
except ImportError:
    EndpointerMode = None

load_dotenv()
SERVER_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8765")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-es-0.42")
VOSK_FIN_SILENCIO = float(os.getenv("VOSK_FIN_SILENCIO", "1.2"))
VOSK_MAX_FRASE = float(os.getenv("VOSK_MAX_FRASE", "30"))
DEBUG = os.getenv("DEBUG_STT", "1") == "0"

SAMPLE_RATE = 16000
BLOCK_MS = 100
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000          # 1600 muestras
MAX_BUFFER = SAMPLE_RATE * 2 * 40                    # 40 s de audio int16
ESPERA_PREGUNTA_S = 6.0                              # tras decir solo "dani"
ESPERA_MIENTRAS_HABLA_S = 10.0                       # si está hablando, no cortar

ROBOT_RECENT_TEXTS = []
WAKE_WORDS_MAP = {
    "robots": ["robots", "robot"],
    "dani": ["dani", "Dani", "robot dani"],
    "gabi": ["gabi", "Gabi", "robot gabi", "gaby", "Gaby", "robot gaby"],
}


# ==============================================================================
# UTILIDADES DE TEXTO
# ==============================================================================
def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s]", "", texto)
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


def separar_wake_word(texto: str):
    """Devuelve (target, resto). Comprueba dani/gabi antes que el genérico 'robots'."""
    for target in ("dani", "gabi", "robots"):
        for alias in sorted(WAKE_WORDS_MAP[target], key=len, reverse=True):
            m = re.search(rf"\b{re.escape(alias)}\b[\s,.:;!?¡¿-]*", texto, flags=re.IGNORECASE)
            if m:
                resto = (texto[:m.start()] + " " + texto[m.end():]).strip()
                return target, resto
    return None, texto


# ==============================================================================
# RECONOCIMIENTO DE HABLANTE (Resemblyzer)
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
        "Alex": ["voices/alex_ref.wav"],
        "Agustin": ["voices/agustin_ref.wav"],
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


def reconocer_hablante(pcm: bytes) -> tuple[str, float]:
    """pcm: audio int16 mono a 16 kHz (el mismo que ve Vosk)."""
    try:
        wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        wav = preprocess_wav(wav, source_sr=SAMPLE_RATE)
        firma_actual = encoder.embed_utterance(wav)

        mejor_match, max_similitud = "Desconocido", 0.0
        UMBRAL_CERTEZA = 0.75
        for nombre, firma_conocida in VOCES_CONOCIDAS.items():
            norm_a = np.linalg.norm(firma_actual)
            norm_b = np.linalg.norm(firma_conocida)
            if norm_a > 0 and norm_b > 0:
                similitud = np.dot(firma_actual, firma_conocida) / (norm_a * norm_b)
                if similitud > max_similitud:
                    max_similitud, mejor_match = similitud, nombre

        porcentaje = float(max_similitud * 100)
        if max_similitud < UMBRAL_CERTEZA:
            return "Desconocido", porcentaje
        return mejor_match, porcentaje
    except Exception:
        return "Desconocido", 0.0


# ==============================================================================
# WHISPER OPCIONAL (solo para refinar el texto final, no para el wake word)
# ==============================================================================
WHISPER_MODEL = None
if os.getenv("USE_WHISPER_FINAL", "0") == "1":
    from faster_whisper import WhisperModel

    print("Cargando faster-whisper...")
    WHISPER_MODEL = WhisperModel(os.getenv("WHISPER_SIZE", "small"), device="auto", compute_type="int8")


def transcribir_whisper(pcm: bytes) -> str:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segmentos, _ = WHISPER_MODEL.transcribe(
        audio, language="es", beam_size=1, vad_filter=True, condition_on_previous_text=False
    )
    return " ".join(s.text.strip() for s in segmentos).strip()


async def procesar_frase(audio: bytes, texto_vosk: str):
    """Identifica al hablante y, si Whisper está activo, mejora el texto (en paralelo)."""
    if WHISPER_MODEL is None:
        speaker, certeza = await asyncio.to_thread(reconocer_hablante, audio)
        return texto_vosk, speaker, certeza
    texto_w, (speaker, certeza) = await asyncio.gather(
        asyncio.to_thread(transcribir_whisper, audio),
        asyncio.to_thread(reconocer_hablante, audio),
    )
    return (texto_w or texto_vosk), speaker, certeza


# ==============================================================================
# COMUNICACIÓN CON EL SERVIDOR (una sola conexión reutilizable)
# ==============================================================================
class Emisor:
    def __init__(self):
        self.ws = None

    async def enviar(self, payload: dict):
        for _ in range(2):
            try:
                if self.ws is None:
                    self.ws = await websockets.connect(SERVER_URI)
                await self.ws.send(json.dumps(payload))
                return
            except Exception:
                self.ws = None
        print("⚠️ No se pudo enviar el evento al servidor.")


emisor = Emisor()


async def enviar_input(target, texto, speaker, certeza):
    conocido = speaker != "Desconocido"
    await emisor.enviar({
        "type": "HUMAN_INPUT",
        "speaker": speaker if conocido else "Persona",
        "is_known_speaker": conocido,
        "confidence": round(float(certeza), 2),
        "target": target,
        "text": texto,
    })


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
# MÁQUINA DE ESTADOS
# ==============================================================================
IDLE = "IDLE"
ESPERANDO = "ESPERANDO_PREGUNTA"


class Sesion:
    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.estado = IDLE
        self.target = None
        self.aviso = False        # ya mandamos HUMAN_LISTENING_START
        self.deadline = 0.0


async def cancelar_aviso(s: Sesion):
    """Falso positivo: devolvemos el robot a reposo si ya le habíamos avisado."""
    if s.aviso and s.target:
        await emisor.enviar({"type": "HUMAN_LISTENING_TIMEOUT", "target": s.target})
        print("↩️  Falso positivo, volviendo a reposo.")
    s.reiniciar()


async def manejar_parcial(s: Sesion, parcial: str):
    if not parcial:
        return
    if DEBUG:
        print(f"   … {parcial}")

    if s.estado == ESPERANDO:
        s.deadline = time.monotonic() + ESPERA_MIENTRAS_HABLA_S  # está hablando
        return

    if not s.aviso:
        target, _ = separar_wake_word(parcial)
        if target and not es_eco_del_robot(parcial):
            s.target, s.aviso = target, True
            print(f"⚡ Wake word en parcial: \"{parcial}\" -> LISTENING ({target.upper()})")
            await emisor.enviar({"type": "HUMAN_LISTENING_START", "target": target})


async def manejar_final(s: Sesion, texto: str, audio: bytes, t_final: float):
    if s.estado == IDLE:
        if not texto or es_eco_del_robot(texto):
            await cancelar_aviso(s)
            return

        target, resto = separar_wake_word(texto)
        if not target:
            await cancelar_aviso(s)
            return

        # CASO 1: "dani, yo opino que..." -> directo, sin esperar otra frase
        if len(resto.split()) >= 3:
            txt, speaker, certeza = await procesar_frase(audio, texto)
            t2, resto2 = separar_wake_word(txt)
            resto_final = resto2 if t2 else resto
            await enviar_input(target, resto_final, speaker, certeza)
            ms = (time.perf_counter() - t_final) * 1000
            print(f"🚀 Directo [{speaker}] \"{resto_final}\"  (fin de frase -> envío: {ms:.0f} ms)")
            s.reiniciar()
            return

        # CASO 2: solo el nombre -> pantalla de escucha y esperar la pregunta
        if not s.aviso:
            await emisor.enviar({"type": "HUMAN_LISTENING_START", "target": target})
        s.estado, s.target, s.aviso = ESPERANDO, target, True
        s.deadline = time.monotonic() + ESPERA_PREGUNTA_S
        print(f"🎤 Solo el nombre. Escuchando la pregunta para {target.upper()}...")
        return

    # estado ESPERANDO: esta frase es la pregunta
    if not texto or es_eco_del_robot(texto):
        return
    txt, speaker, certeza = await procesar_frase(audio, texto)
    t3, r3 = separar_wake_word(txt)
    pregunta = (r3 if t3 else txt).strip()
    if not pregunta:
        return
    await enviar_input(s.target, pregunta, speaker, certeza)
    ms = (time.perf_counter() - t_final) * 1000
    print(f"🚀 Pregunta [{speaker}] \"{pregunta}\"  (fin de frase -> envío: {ms:.0f} ms)")
    s.reiniciar()


def crear_reconocedor() -> KaldiRecognizer:
    SetLogLevel(-1)
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    if EndpointerMode is not None:
        try:
            rec.SetEndpointerMode(EndpointerMode.SHORT)
            # (silencio inicial máx., silencio que cierra frase, duración máx. de frase)
            rec.SetEndpointerDelays(5.0, VOSK_FIN_SILENCIO, VOSK_MAX_FRASE)
        except Exception as e:
            print(f"ℹ️ No se pudo ajustar el endpointer: {e}")
    return rec


async def flujo_escucha_principal():
    loop = asyncio.get_running_loop()
    cola: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames, t, status):
        loop.call_soon_threadsafe(cola.put_nowait, bytes(indata))

    print(f"Cargando modelo Vosk ({VOSK_MODEL_PATH})...")
    rec = crear_reconocedor()
    s = Sesion()
    buffer = bytearray()
    mic_index = os.getenv("MIC_INDEX")
    dispositivo = int(mic_index) if mic_index else None

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                           channels=1, device=dispositivo, callback=callback):
        print("\n[PASO 1] 👂 Esperando palabra clave (dani / gabi)...")
        while True:
            try:
                data = await cola.get()

                buffer.extend(data)
                if len(buffer) > MAX_BUFFER:
                    del buffer[:-MAX_BUFFER]

                if rec.AcceptWaveform(data):
                    texto = json.loads(rec.Result()).get("text", "").strip()
                    audio = bytes(buffer)
                    buffer.clear()
                    await manejar_final(s, texto, audio, time.perf_counter())
                else:
                    parcial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    await manejar_parcial(s, parcial)

                # Se dijo solo el nombre y nadie ha hablado después
                if s.estado == ESPERANDO and time.monotonic() > s.deadline:
                    print("⏱️ Tiempo de espera agotado sin pregunta. Volviendo a reposo...")
                    await emisor.enviar({"type": "HUMAN_LISTENING_TIMEOUT", "target": s.target})
                    rec.Reset()
                    buffer.clear()
                    s.reiniciar()
                    print("\n[PASO 1] 👂 Esperando palabra clave (dani / gabi)...")

            except Exception as e:
                print(f"⚠️ {e}")
                await asyncio.sleep(0.05)


async def main():
    await asyncio.gather(listen_server_broadcasts(), flujo_escucha_principal())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CAPTURADOR] Detenido.")