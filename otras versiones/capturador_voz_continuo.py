import asyncio
import json
import queue
import threading
import websockets
import speech_recognition as sr
from difflib import SequenceMatcher

SERVER_URI = "ws://localhost:8765"

# Memoria local de frases pronunciadas por los robots (Filtro Anti-Eco)
ROBOT_RECENT_TEXTS = []

# Cola de mensajes para procesar audio sin bloquear el micrófono
AUDIO_QUEUE = queue.Queue()

def is_similar_to_robot_speech(text: str, threshold: float = 0.45) -> bool:
    """Compara si la frase capturada se parece a algo dicho recientemente por el robot."""
    text_clean = text.lower().strip()
    for robot_phrase in ROBOT_RECENT_TEXTS:
        rf_clean = robot_phrase.lower().strip()
        similarity = SequenceMatcher(None, text_clean, rf_clean).ratio()
        
        # Coincidencia por porcentaje o inclusión de fragmentos significativos
        if similarity >= threshold or (len(text_clean) > 6 and text_clean in rf_clean) or (len(rf_clean) > 6 and rf_clean in text_clean):
            return True
    return False

async def listen_robot_broadcaster():
    """Recibe en tiempo real lo que dicen los robots para alimentar el filtro anti-eco."""
    while True:
        try:
            async with websockets.connect(SERVER_URI) as ws:
                await ws.send(json.dumps({"type": "REGISTER_CAPTURER"}))
                print("🔗 [CAPTURADOR] Sincronización anti-eco conectada con el servidor.")
                
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "ROBOT_SPOKE":
                        phrase = data.get("text", "")
                        if phrase:
                            ROBOT_RECENT_TEXTS.append(phrase)
                            if len(ROBOT_RECENT_TEXTS) > 8:
                                ROBOT_RECENT_TEXTS.pop(0)
                            print(f"🤖 [SYNC] Guardado en anti-eco: \"{phrase[:35]}...\"")
        except Exception:
            await asyncio.sleep(2)

def background_mic_worker(recognizer, mic):
    """Captura continua de audio sin bloquear el procesamiento."""
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                if audio:
                    AUDIO_QUEUE.put(audio)
        except Exception:
            pass

async def process_audio_queue():
    """Transcribe el audio y aplica filtros antes de enviarlo al servidor."""
    recognizer = sr.Recognizer()
    
    while True:
        if not AUDIO_QUEUE.empty():
            audio = AUDIO_QUEUE.get()
            
            try:
                raw_text = await asyncio.to_thread(
                    recognizer.recognize_google, audio, language="es-ES"
                )
                
                clean_text = raw_text.strip()
                if clean_text:
                    # Filtro 1: Ignorar ruidos o interjecciones cortas que rompen la frase
                    words = clean_text.split()
                    if len(words) < 2 and len(clean_text) < 5:
                        print(f"🤫 [IGNORADO] Frase demasiado corta o ruido: \"{clean_text}\"")
                        continue

                    # Filtro 2: Anti-eco del robot
                    if is_similar_to_robot_speech(clean_text):
                        print(f"🤖 🛑 [ANTI-ECO] Audio del altavoz descartado: \"{clean_text[:40]}...\"")
                        continue

                    print(f"👉 🎤 [HUMANO]: \"{clean_text}\"")
                    
                    # Enviar al servidor orquestador
                    async with websockets.connect(SERVER_URI) as ws:
                        payload = {
                            "type": "HUMAN_INPUT",
                            "speaker": "H1",
                            "text": clean_text
                        }
                        await ws.send(json.dumps(payload))
                        print("  ⚡ [ENVIADO AL ORQUESTADOR]")

            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"[!] Error procesando audio: {e}")
        
        await asyncio.sleep(0.05)

async def main():
    recognizer = sr.Recognizer()
    
    # --- AJUSTES DE TIEMPOS EQUILIBRADOS PARA PAUSAS AL HABLAR ---
    recognizer.pause_threshold = 1.4        # Subimos a 1.4s para permitir pausas al hablar sin cortar
    recognizer.non_speaking_duration = 0.6
    recognizer.energy_threshold = 450        # Sensibilidad ajustada para evitar capturar respiraciones
    recognizer.dynamic_energy_threshold = True

    mic = sr.Microphone()

    print("\n[MIC] Calibrando silencio de la sala (1 segundo)...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("[MIC] ¡Micrófono activado con margen de pausa de 1.4s!\n")

    # Iniciar captura en hilo de fondo
    threading.Thread(target=background_mic_worker, args=(recognizer, mic), daemon=True).start()

    # Iniciar tareas asíncronas
    await asyncio.gather(
        process_audio_queue(),
        listen_robot_broadcaster()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[MIC] Capturador detenido.")