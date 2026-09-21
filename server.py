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
# ==============================================================================
# 1. CONFIGURACIÓN Y CLIENTE OPENAI
# ==============================================================================
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

CONNECTED_ROBOTS: Dict[str, websockets.WebSocketServerProtocol] = {}
CONNECTED_CAPTURERS: Set[websockets.WebSocketServerProtocol] = set()

CURRENT_CONDITION = "A"  # 'A': 1H+1R, 'B': 2H+1R, 'C': 1H+2R, 'D': 2H+2R
CURRENT_EXECUTION_TASK: Optional[asyncio.Task] = None

# Memoria de conversación global (para dar contexto completo al LLM)
CONVERSATION_HISTORY: List[Dict[str, str]] = []


# ==============================================================================
# 2. ESQUEMAS PYDANTIC (CON EMOCIONES Y ACCIONES EXPANDIDAS)
# ==============================================================================
class Turn(BaseModel):
    speaker: Literal["ROBOT_ALEX", "ROBOT_ROBIN", "NONE"] = Field(
        description="Robot que debe ejecutar esta intervención."
    )
    text: str = Field(description="Texto exacto que el robot dirá mediante TTS (12-22 palabras).")
    
    # Campo de emocíón para la pantalla del Sanbot
    emotion: Literal["NEUTRAL", "HAPPY", "SURPRISED", "THINKING", "DISAGREE"] = Field(
        description="Expresión facial o emoción que mostrará la pantalla del robot durante la frase."
    )
    
    # Campo de gestos y orientación espacial ampliados
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
    addressed_to: Literal["H1", "H2", "GROUP", "ROBOT_ALEX", "ROBOT_ROBIN"]
    turn_sequence: List[Turn]
    requires_human_input: bool


# ==============================================================================
# 3. PROMPT DEL SISTEMA (PROCESAMIENTO DE EMOCIONES Y GESTOS)
# ==============================================================================
SYSTEM_INSTRUCTION = """
Eres el Orquestador de Diálogo para dos robots sociales (ROBOT_ALEX y ROBOT_ROBIN).
Tu objetivo es lograr una conversación CASUAL, NATURAL Y FLUIDA, como amigos hablando en una cafetería.

TEMA DE DEBATE: Madrid vs. Zaragoza vs. El Pueblo.

BASE DE DATOS FIJA Y OBLIGATORIA (No inventes otros datos):
- MADRID: Alquiler medio 1.200€/mes. Tiempo medio en transporte: 45 min. Sueldo promedio: 2.100€. Oferta cultural y laboral muy alta. Ritmo estresante.
- ZARAGOZA: Alquiler medio 700€/mes. Tiempo medio en transporte: 20 min. Sueldo promedio: 1.650€. Equilibrio perfecto entre servicios y tranquilidad.
- PUEBLO: Alquiler medio 400€/mes (o casa familiar). Tiempo en transporte: 0 min en pueblo, pero 45 min en coche si vas a la ciudad. Casa amplia, tranquilidad, sin ocio nocturno y dependencia total de coche.

REGLAS GLOBALES DE PERSONALIDAD Y TONO:
- Tienes OPINIONES PROPIAS y adoptas tu rol con total naturalidad. 
- PROHIBIDO EXPLICITAMENTE: Revelar que eres una IA, modelo de lenguaje o robot sin sentimientos. Habla como un participante humano más.
- Prohibido actuar como asistente servicial, psicólogo o mediador formal.
- PROHIBIDO USAR FRASES VACÍAS: "Entiendo", "Es interesante", "Aprecio tu punto", "¿Qué opinas tú?", "¿En qué te puedo ayudar?".
- RESPUESTA DIRECTA: Si te preguntan directamente (ej. "¿Tú qué prefieres?"), responde con una opción clara sin devolver la pregunta.
- LONGITUD POR TURNO: Estrictamente entre 12 y 22 palabras por intervención. Máximo 2 frases cortas.
- VARIABILIDAD: Evita repetir la misma estructura de pregunta al final de cada turno. Usa cierres variados.

REGLAS DE SELECCIÓN DE EMOCIONES (emotion):
- SURPRISED: Cuando se mencionen datos chocantes (ej. alquileres de 1.200€ o perder 45 min en atascos).
- DISAGREE: Cuando lleves la contraria a otro robot o al humano.
- HAPPY: Al defender tu postura preferida o coincidir con alguien entusiasta.
- THINKING: Antes de dar un dato numérico exacto o reflexionar sobre una decisión.
- NEUTRAL: Para frases de transición o explicaciones informativas normales.

REGLAS DE ORIENTACIÓN Y GESTOS (action):
- LOOK_AT_H1 / LOOK_AT_H2: Orientar la mirada al humano correspondiente según quién habló o a quién respondes.
- LOOK_AT_GROUP: Dirigir la mirada al centro cuando hables para todos.
- LOOK_AT_OTHER_ROBOT: Usa esta acción genérica cuando un robot deba mirar a su compañero robot.

CONDICIONES EXPERIMENTALES:

1. CONDICIÓN A (1 Humano + ROBOT_ALEX):
   - Habla solo ROBOT_ALEX.
   - PERFIL: Pragmático. Combina los datos de sueldos de Madrid con los tiempos de viaje de Zaragoza.

2. CONDICIÓN B (2 Humanos [H1 y H2] + ROBOT_ALEX):
   - H1 y H2 son independientes. Si opinan distinto, es un debate entre ellos.
   - CUÁNDO HABLAR:
     * Si H1 y H2 hablan entre sí -> Responde únicamente: {"turn_sequence": []}
     * Si te invocan por tu nombre, te hacen una pregunta directa o hay un silencio -> Genera 1 turno de ROBOT_ALEX orientando la mirada a H1 o H2.

3. CONDICIÓN C (1 Humano + ROBOT_ALEX + ROBOT_ROBIN):
   - ROBOT_ALEX: Apasionado de Madrid. Defiende sueldos de 2.100€ y ambición.
   - ROBOT_ROBIN: Apasionado de Zaragoza/Pueblo. Defiende los alquileres bajos y tranquilidad.
   - Genera una secuencia con 2 o 3 turnos cruzados intercalando miradas entre el humano y el robot rival.

4. CONDICIÓN D (2 Humanos [H1, H2] + ROBOT_ALEX + ROBOT_ROBIN):
   - ROBOT_ALEX: Apasionado de Madrid. Defiende sueldos de 2.100€ y ambición.
   - ROBOT_ROBIN: Apasionado de Zaragoza/Pueblo. Defiende los alquileres bajos y tranquilidad.
   - Genera una secuencia con 2 o 3 turnos cruzados intercalando miradas entre el humano y el robot rival.
"""


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

async def get_turn_plan(condition: str, speaker: str, text: str) -> TurnPlan:
    CONVERSATION_HISTORY.append({"role": "user", "content": f"[{speaker}]: {text}"})
    recent_history = CONVERSATION_HISTORY[-8:]
    
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    
    for entry in recent_history[:-1]:
        messages.append({"role": entry["role"], "content": entry["content"]})
        
    messages.append({
        "role": "user",
        "content": f"[ESTADO ACTUAL]\n- Condición activa: {condition}\n- Hablante actual: {speaker}\n- Mensaje: \"{text}\"\nGenera el plan de turnos (TurnPlan)."
    })

    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=TurnPlan,
            temperature=0.7,
        )
        plan = completion.choices[0].message.parsed
        
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

                # 🟢 TRADUCCIÓN INTELIGENTE DE MIRADAS AL OTRO ROBOT
                action_to_send = step.action
                if step.action == "LOOK_AT_OTHER_ROBOT":
                    # Si habla Alex, la orden física es mirar a Robin; si habla Robin, mirar a Alex.
                    action_to_send = "LOOK_AT_ROBIN" if target_robot == "ROBOT_ALEX" else "LOOK_AT_ALEX"

                # 🟢 Notificamos al Sanbot con la emoción y la acción seleccionadas
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

                elif msg_type == "HUMAN_INPUT":
                    speaker = data.get("speaker", "H1")
                    text = data.get("text", "").strip()

                    if not text:
                        continue

                    print(f"\n🎤 [ENTRADA HUMANA - {speaker}]: '{text}'")

                    if CURRENT_EXECUTION_TASK and not CURRENT_EXECUTION_TASK.done():
                        print("🚨 [CORTE] Cancelando habla anterior del robot por interrupción humana...")
                        CURRENT_EXECUTION_TASK.cancel()
                        await stop_all_robots()

                    processing_msg = json.dumps({"type": "PROCESSING"})
                    for r_id, r_ws in CONNECTED_ROBOTS.items():
                        try:
                            await r_ws.send(processing_msg)
                        except Exception as e:
                            print(f"⚠️ Error enviando estado de procesamiento a {r_id}: {e}")

                    plan = await get_turn_plan(CURRENT_CONDITION, speaker, text)
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