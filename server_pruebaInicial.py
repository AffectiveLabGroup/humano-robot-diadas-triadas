import asyncio
import json
import os
from typing import Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field
import websockets
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env
load_dotenv()

# Recupera la clave desde las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ No se encontró la API Key en el entorno o en el archivo .env")

# ==============================================================================
# 1. CONFIGURACIÓN Y CLIENTE OPENAI
# ==============================================================================
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

CONNECTED_ROBOTS: Dict[str, websockets.WebSocketServerProtocol] = {}
CONNECTED_CAPTURERS: Set[websockets.WebSocketServerProtocol] = set()

CURRENT_CONDITION = "A"  # 'A': 1H+1R, 'B': 2H+1R, 'C': 1H+2R, 'D': 2H+2R
CURRENT_EXECUTION_TASK: Optional[asyncio.Task] = None

# Memoria de conversación global
CONVERSATION_HISTORY: List[Dict[str, str]] = []


# ==============================================================================
# 2. ESQUEMAS PYDANTIC
# ==============================================================================
class Turn(BaseModel):
    speaker: Literal["ROBOT_DANI", "ROBOT_CRIS", "NONE"] = Field(
        description="Robot que debe ejecutar esta intervención."
    )
    text: str = Field(description="Texto exacto que el robot dirá mediante TTS (12-22 palabras).")
    
    emotion: Literal["NEUTRAL", "HAPPY", "SURPRISED", "THINKING", "DISAGREE"] = Field(
        description="Expresión facial o emoción que mostrará la pantalla del robot durante la frase."
    )
    
    action: Literal[
        "LOOK_AT_H1", "LOOK_AT_H2", "LOOK_AT_GROUP", 
        "LOOK_AT_OTHER_ROBOT", "RAISE_ARMS", "NOD_HEAD", "IDLE"
    ] = Field(
        description="Acción de mirada o gesto. Usa LOOK_AT_OTHER_ROBOT para mirar al compañero robot."
    )
    
    delay_after_ms: int = Field(
        default=500,
        description="Pausa en ms tras terminar la frase antes del siguiente turno."
    )

class TurnPlan(BaseModel):
    condition: Literal["A", "B", "C", "D"]
    addressed_to: Literal["H1", "H2", "GROUP", "ROBOT_DANI", "ROBOT_CRIS"]
    turn_sequence: List[Turn]
    requires_human_input: bool


# ==============================================================================
# 3. PROMPT DEL SISTEMA Y REGLAS DE CONDICIONES
# ==============================================================================
BASE_SYSTEM_PROMPT = """
Tu objetivo es gestionar una conversación CASUAL, NATURAL y MODERADA.

REGLAS DE DIRECCIÓN Y TURNOS:
1. SI EL MENSAJE INDICA UN 'ROBOT DESTINATARIO', ESE ROBOT DEBE SER EL PRIMERO EN ENTRAR EN turn_sequence.
2. Prohibido revelar que eres IA. Sin frases vacías ("Entiendo", "Aprecio tu punto", "Es interesante"). 
3. NUNCA pronuncies las palabras literales "Persona", "H1" o "H2". Si el hablante no tiene nombre conocido, dirígete a él/ella como simplemente "tú", o incluye frases estilo "¿Qué opinas sobre lo que ha dicho?".
4. Responde directo. Longitud por turno: 22 a 32 palabras por intervención.
5. Emociones válidas: SURPRISED, DISAGREE, HAPPY, THINKING, NEUTRAL.
6. Acciones válidas: LOOK_AT_H1, LOOK_AT_H2, LOOK_AT_GROUP, LOOK_AT_OTHER_ROBOT, RAISE_ARMS, NOD_HEAD, IDLE.
"""

CONDITION_RULES = {
    "A": "CONDICIÓN A (1 Humano + ROBOT_DANI): Habla solo ROBOT_DANI. PERFIL: Pragmático.",
}

# ==============================================================================
# 4. FUNCIONES DE DIFUSIÓN E INTERRUPCIÓN
# ==============================================================================
async def broadcast_robot_speech_to_capturers(text: str):
    if CONNECTED_CAPTURERS:
        payload = json.dumps({"type": "ROBOT_SPOKE", "text": text})
        for ws in list(CONNECTED_CAPTURERS):
            try:
                await ws.send(payload)
            except Exception:
                CONNECTED_CAPTURERS.remove(ws)

async def stop_all_robots():
    payload = json.dumps({"type": "STOP_SPEECH"})
    for robot_id, ws in CONNECTED_ROBOTS.items():
        try:
            await ws.send(payload)
            print(f"🛑 [INTERRUPCIÓN] Enviada orden de parar voz a {robot_id}")
        except Exception:
            pass

async def get_turn_plan(condition: str, speaker: str, text: str, target_robot: Optional[str] = None) -> TurnPlan:
    # Registramos la entrada en la memoria general
    target_str = f" | Dirigido a: ROBOT_{target_robot.upper()}" if target_robot else ""
    CONVERSATION_HISTORY.append({"role": "user", "content": f"[{speaker}{target_str}]: {text}"})
    
    # Aumentamos a los últimos 6 mensajes para que no pierda la memoria del otro interlocutor
    recent_history = CONVERSATION_HISTORY[-6:]
    
    # CONSTRUCCIÓN DEL PROMPT CON CONTEXTO CLARO DE MESA
    robots_en_sala = "SOLO ROBOT_DANI (CRIS NO EXISTE)" if condition in ['A', 'B'] else "ROBOT_DANI y ROBOT_CRIS"
    participantes_humanos = "1 Humano (H1)" if condition in ['A', 'C'] else "2 Humanos en la mesa (H1 y H2)"

    system_content = f"""{BASE_SYSTEM_PROMPT}

======================================================================
CONFIGURACIÓN DE LA SESIÓN ACTUAL:
- CONDICIÓN: [{condition}]
- PARTICIPANTES HUMANOS EN MESA: {participantes_humanos}
- ROBOTS DISPONIBLES EN SALA: {robots_en_sala}
======================================================================
REGLAS ESPECÍFICAS DE LA CONDICIÓN [{condition}]:
{CONDITION_RULES.get(condition, '')}
"""
    
    messages = [{"role": "system", "content": system_content}]
    for entry in recent_history[:-1]:
        messages.append({"role": entry["role"], "content": entry["content"]})
        
    prompt_user = f"Hablante actual: {speaker}{target_str} | Mensaje: \"{text}\"\nGenera el TurnPlan en JSON."
    messages.append({
        "role": "user",
        "content": prompt_user
    })

    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=TurnPlan,
            temperature=0.3,
            max_tokens=250
        )
        
        plan = completion.choices[0].message.parsed
        
        # Guardamos la respuesta del robot en el historial
        for turn in plan.turn_sequence:
            if turn.speaker != "NONE":
                CONVERSATION_HISTORY.append({
                    "role": "assistant", 
                    "content": f"[{turn.speaker}]: {turn.text}"
                })
                
        return plan

    except Exception as e:
        print(f"❌ Error API OpenAI: {e}")
        return TurnPlan(
            condition=condition, # type: ignore
            addressed_to="GROUP",
            turn_sequence=[],
            requires_human_input=True
        )


# ==============================================================================
# 5. EJECUCIÓN DE TURNOS
# ==============================================================================
async def execute_turn_plan(plan: TurnPlan):
    try:
        if len(plan.turn_sequence) == 0:
            print("🤫 [ORQUESTADOR] Charla entre humanos o sin intervención. Robot en silencio.\n")
            return

        for step in plan.turn_sequence:
            target_robot = step.speaker

            if target_robot in CONNECTED_ROBOTS:
                await broadcast_robot_speech_to_capturers(step.text)

                action_to_send = step.action
                if step.action == "LOOK_AT_OTHER_ROBOT":
                    action_to_send = "LOOK_AT_CRIS" if target_robot == "ROBOT_DANI" else "LOOK_AT_DANI"

                payload = {
                    "type": "EXECUTE_TURN",
                    "text": step.text,
                    "emotion": step.emotion,
                    "action": action_to_send
                }
                print(f" -> [{target_robot}] Emotion: {step.emotion} | Action: {action_to_send} | Text: \"{step.text}\"")

                await CONNECTED_ROBOTS[target_robot].send(json.dumps(payload))

                speaking_duration = (len(step.text) / 14.0) + 0.8
                total_wait = speaking_duration + (step.delay_after_ms / 1000.0)
                
                await asyncio.sleep(total_wait)
            else:
                print(f" ⚠️ [!] {target_robot} no está conectado.")

        print("[ORQUESTADOR] Fin de la secuencia de diálogo.\n")

    except asyncio.CancelledError:
        print("⚡ [INTERRUPCIÓN] Secuencia cancelada por nueva entrada de voz humana.")
        raise


# ==============================================================================
# 6 Y 7. MANEJO DE WEBSOCKETS E INICIO
# ==============================================================================
async def handler(websocket):
    global CURRENT_EXECUTION_TASK, CURRENT_CONDITION
    client_id = None
    is_capturer = False

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "REGISTER":
                    client_id = data.get("robot_id")
                    if "condition" in data:
                        cond_raw = data.get("condition")
                        if "Condicion_A" in cond_raw or cond_raw == "A": CURRENT_CONDITION = "A"
                        elif "Condicion_B" in cond_raw or cond_raw == "B": CURRENT_CONDITION = "B"
                        elif "Condicion_C" in cond_raw or cond_raw == "C": CURRENT_CONDITION = "C"
                        elif "Condicion_D" in cond_raw or cond_raw == "D": CURRENT_CONDITION = "D"

                    CONNECTED_ROBOTS[client_id] = websocket
                    print(f"[CONEXIÓN] Robot {client_id} conectado. Condición asignada: [{CURRENT_CONDITION}]")
                    await websocket.send(json.dumps({"status": "REGISTERED", "robot_id": client_id}))

                elif msg_type == "REGISTER_CAPTURER":
                    is_capturer = True
                    CONNECTED_CAPTURERS.add(websocket)
                    print("🔗 [CONEXIÓN] Capturador de audio registrado para filtro anti-eco.")

                elif msg_type == "HUMAN_LISTENING_START":
                    target = data.get("target")
                    speaker = data.get("speaker", "Humano")
                    print(f"👂 [ESTADO] Escuchando a {speaker}...")
                    # Reenviamos la orden al Sanbot para que cambie la pantalla a modo "Escuchando"
                    listening_msg = json.dumps({"type": "LISTENING", "speaker": speaker, "target": target})
                    for r_id, r_ws in CONNECTED_ROBOTS.items():
                        await r_ws.send(listening_msg)

                elif msg_type == "HUMAN_INPUT":
                    speaker = data.get("speaker", "H1")
                    text = data.get("text", "").strip()
                    target = data.get("target")

                    if not text:
                        continue

                    log_target = f" -> Dirigido a {target.upper()}" if target else ""
                    print(f"\n🎤 [ENTRADA HUMANA - {speaker}{log_target}]: '{text}'")

                    if CURRENT_EXECUTION_TASK and not CURRENT_EXECUTION_TASK.done():
                        print("🚨 [CORTE] Cancelando habla anterior del robot por interrupción humana...")
                        CURRENT_EXECUTION_TASK.cancel()
                        await stop_all_robots()

                    processing_msg = json.dumps({"type": "PROCESSING", "target": target})
                    for r_id, r_ws in CONNECTED_ROBOTS.items():
                        try:
                            await r_ws.send(processing_msg)
                        except Exception as e:
                            print(f"⚠️ Error enviando estado de procesamiento a {r_id}: {e}")

                    plan = await get_turn_plan(CURRENT_CONDITION, speaker, text, target_robot=target)
                    CURRENT_EXECUTION_TASK = asyncio.create_task(execute_turn_plan(plan))

            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"⚠️ Error procesando mensaje: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if client_id and client_id in CONNECTED_ROBOTS:
            del CONNECTED_ROBOTS[client_id]
            print(f"[DESCONEXIÓN] Robot {client_id} desconectado.")
        if is_capturer and websocket in CONNECTED_CAPTURERS:
            CONNECTED_CAPTURERS.remove(websocket)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("==================================================")
        print(" SERVIDOR ORQUESTADOR ACTIVO (ws://localhost:8765)")
        print(f" Condición activa inicial: [{CURRENT_CONDITION}]")
        print("==================================================")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SERVIDOR] Detenido.")