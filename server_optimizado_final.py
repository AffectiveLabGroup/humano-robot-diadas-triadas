import asyncio
import json
import os
from typing import Dict, List, Literal, Optional, Set
from pydantic import BaseModel, Field
import websockets
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ No se encontró la API Key en el entorno o en el archivo .env")

# ==============================================================================
# 1. CONFIGURACIÓN Y CLIENTE OPENAI
# ==============================================================================
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

CONNECTED_ROBOTS: Dict[str, websockets.WebSocketServerProtocol] = {}
CONNECTED_CAPTURERS: Set[websockets.WebSocketServerProtocol] = set()

# Un evento por robot: el robot lo activa (TURN_DONE) cuando termina de hablar
TURN_DONE_EVENTS: Dict[str, asyncio.Event] = {}

CURRENT_CONDITION = "A"  # 'A': 1H+1R, 'B': 2H+1R, 'C': 1H+2R, 'D': 2H+2R
CURRENT_EXECUTION_TASK: Optional[asyncio.Task] = None

CONVERSATION_HISTORY: List[Dict[str, str]] = []


# ==============================================================================
# 2. ESQUEMAS PYDANTIC
# ==============================================================================
class Turn(BaseModel):
    speaker: Literal["ROBOT_DANI", "ROBOT_GABI", "NONE"] = Field(
        description="Robot que debe ejecutar esta intervención."
    )
    text: str = Field(description="Texto exacto que el robot dirá mediante TTS (22-32 palabras).")

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
    addressed_to: Literal["H1", "H2", "GROUP", "ROBOT_DANI", "ROBOT_GABI"]
    turn_sequence: List[Turn]
    requires_human_input: bool


# ==============================================================================
# 3. PROMPT DEL SISTEMA Y REGLAS DE CONDICIONES
# ==============================================================================
BASE_SYSTEM_PROMPT = """
Eres el Orquestador de Diálogo para dos robots sociales (ROBOT_DANI y ROBOT_GABI).
Tu objetivo es gestionar una conversación CASUAL, NATURAL y MODERADA para decidir la ubicación ideal para vivir (Madrid, Zaragoza o Pueblo).

DATOS OBLIGATORIOS DE OPINIÓN:
- MADRID: Alquiler aproximado 1.200€, Transporte 45 min, Sueldo 2.100€. Oferta cultural alta, estresante.
- ZARAGOZA: Alquiler aproximado 700€, Transporte 20 min, Sueldo 1.650€. Equilibrio.
- PUEBLO: Alquiler aproximado 400€, Transporte 0 min en pueblo (45 en coche). Tranquilidad, sin ocio.

REGLAS DE DIRECCIÓN Y TURNOS:
1. SI EL MENSAJE INDICA UN 'ROBOT DESTINATARIO', ESE ROBOT DEBE SER EL PRIMERO EN ENTRAR EN turn_sequence.
2. Prohibido revelar que eres IA. Sin frases vacías ("Entiendo", "Aprecio tu punto", "Es interesante"). 
3. NUNCA pronuncies las palabras literales "Persona", "H1" o "H2". Si el hablante no tiene nombre conocido, dirígete a él/ella como simplemente "tú", o incluye frases estilo "¿Qué opinas sobre lo que ha dicho?".
4. Responde directo. Longitud por turno: 22 a 32 palabras por intervención.
5. Emociones válidas: SURPRISED, DISAGREE, HAPPY, THINKING, NEUTRAL.
6. Acciones válidas: LOOK_AT_H1, LOOK_AT_H2, LOOK_AT_GROUP, LOOK_AT_OTHER_ROBOT, RAISE_ARMS, NOD_HEAD, IDLE.
7. Si el turno termina con una pregunta al humano, la acción DEBE ser LOOK_AT_H1 (o LOOK_AT_GROUP).
   Usa LOOK_AT_OTHER_ROBOT solo cuando hables directamente al otro robot y él vaya a responder en el siguiente turno.
"""

REGLAS_TURNOS_ROBOTS = """
REGLAS DE TURNOS ENTRE ROBOTS (OBLIGATORIAS):
- Si un robot se dirige al OTRO robot, su frase DEBE empezar con el nombre del compañero ("GABI, ..." / "Dani, ...") y turn_sequence DEBE incluir justo después un turno del compañero respondiéndole.
- Si un robot se dirige al HUMANO, NO nombres al compañero, usa LOOK_AT_H1 (o LOOK_AT_GROUP) y ese debe ser el último turno de la secuencia.
- Nunca dejes una pregunta al compañero sin su respuesta en la misma secuencia.
- Una secuencia termina siempre con una pregunta o comentario dirigido al humano, no al otro robot.
"""

CONDITION_RULES = {
    "A": "CONDICIÓN A (1 Humano + ROBOT_DANI): Habla solo ROBOT_DANI. PERFIL: Pragmático.",

    "B": """CONDICIÓN B (2 Humanos + ROBOT_DANI) - MODERACIÓN SOCIAL:
    - EN ESTA CONDICIÓN SOLO EXISTE ROBOT_DANI. PROHIBIDO NOMBRAR O USAR A GABI.
    - PARTICIPANTES EN MESA: Hay 2 personas humanas. Asocia y memoriza estrictamente lo que dice cada una.
    - REGLA DE INCLUSIÓN GRUPAL (OBLIGATORIA):
    * Al responder a una de las personas, valida su idea en pocas palabras, genera diálogo y LUEGO PREGUNTA A LA OTRA PERSONA su opinión para no dejarla fuera.
    * Ejemplo: Si Carla te habla sobre el teatro en la ciudad, responde a Carla pero termina preguntando: "¿Tú qué opinas Loreto? ¿Que opinas sobre lo que dijo Carla? ¿podrías vivir en un lugar sin teatro?" usando 'LOOK_AT_GROUP' o la mirada hacia la otra persona.
    - Manejo de nombres: Usa el nombre si se conoce por el diálogo. Si no se conoce o figura como 'Persona', 'H1' o 'H2', usa 'tu compañero' o 'tú'. NUNCA inventes nombres.
    - Si los dos humanos hablan exclusivamente entre sí sin invocar al robot, devuelve turn_sequence: [].""",

    "C": ("CONDICIÓN C (1 Humano + DANI + GABI): DANI defiende Madrid. GABI defiende Zaragoza/Pueblo. "
          "Genera 1, 2 o 3 turnos cruzados empezando por el robot invocado.\n" + REGLAS_TURNOS_ROBOTS),

    "D": ("""CONDICIÓN D (2 Humanos + DANI + GABI) - MEDIACIÓN Y DEBATE MULTIPERSONA:
    - Dinámica a 4 bandas: DANI defiende Madrid y GABI defiende Zaragoza/Pueblo.
    - Mediación y Alianzas: Los robots pueden buscar la alianza de los humanos o mediar entre sus posturas para llevarse el debate a su terreno.
    - Orientación física: Usa LOOK_AT_H1 / LOOK_AT_H2 al dirigirse a un humano, LOOK_AT_OTHER_ROBOT cuando los robots hablen entre sí, y LOOK_AT_GROUP al hacer preguntas abiertas.
    - Manejo de nombres: Si es 'Persona', 'H1' o 'H2', usa fórmulas neutras ("tu compañero/a"), NUNCA inventes nombres.
    - Genera 1, 2 o 3 turnos cruzados empezando por el robot invocado.\n""" + REGLAS_TURNOS_ROBOTS)
}

# ==============================================================================
# 4. FUNCIONES DE DIFUSIÓN E INTERRUPCIÓN
# ==============================================================================
def resolver_destinatarios(target: Optional[str]) -> List[str]:
    """'dani' -> ['ROBOT_DANI'], 'GABI' -> ['ROBOT_GABI'].
    'robots', None o un robot no conectado -> todos los conectados."""
    if target:
        robot_id = f"ROBOT_{target.upper()}"
        if robot_id in CONNECTED_ROBOTS:
            return [robot_id]
    return list(CONNECTED_ROBOTS.keys())


async def enviar_a_robots(target: Optional[str], payload: dict):
    msg = json.dumps(payload)
    for robot_id in resolver_destinatarios(target):
        try:
            await CONNECTED_ROBOTS[robot_id].send(msg)
        except Exception as e:
            print(f"⚠️ Error enviando {payload['type']} a {robot_id}: {e}")


async def broadcast_robot_speech_to_capturers(text: str):
    if CONNECTED_CAPTURERS:
        payload = json.dumps({"type": "ROBOT_SPOKE", "text": text})
        for ws in list(CONNECTED_CAPTURERS):
            try:
                await ws.send(payload)
            except Exception:
                CONNECTED_CAPTURERS.discard(ws)


async def stop_all_robots(except_target: Optional[str] = None):
    """Para la voz de todos. Los robots que NO son el destinatario vuelven a reposo."""
    stop_msg = json.dumps({"type": "STOP_SPEECH"})
    idle_msg = json.dumps({"type": "IDLE"})
    protegidos = set(resolver_destinatarios(except_target)) if except_target else set()
    for robot_id, ws in list(CONNECTED_ROBOTS.items()):
        try:
            await ws.send(stop_msg)
            if robot_id not in protegidos:
                await ws.send(idle_msg)
            print(f"🛑 [INTERRUPCIÓN] Enviada orden de parar voz a {robot_id}")
        except Exception:
            pass


async def get_turn_plan(condition: str, speaker: str, text: str, target_robot: Optional[str] = None) -> TurnPlan:
    target_str = f" | Dirigido a: ROBOT_{target_robot.upper()}" if target_robot else ""
    CONVERSATION_HISTORY.append({"role": "user", "content": f"[{speaker}{target_str}]: {text}"})

    recent_history = CONVERSATION_HISTORY[-6:]

    robots_en_sala = "SOLO ROBOT_DANI (GABI NO EXISTE)" if condition in ['A', 'B'] else "ROBOT_DANI y ROBOT_GABI"
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
    messages.append({"role": "user", "content": prompt_user})

    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=TurnPlan,
            temperature=0.3,
            max_tokens=450   # antes 250: evita planes truncados
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
            condition=condition,  # type: ignore
            addressed_to="GROUP",
            turn_sequence=[],
            requires_human_input=True
        )


# ==============================================================================
# 5. VALIDACIÓN Y EJECUCIÓN DE TURNOS
# ==============================================================================
def sanear_plan(plan: TurnPlan) -> TurnPlan:
    """Si un turno dirige la mirada al otro robot pero el siguiente turno no es suyo,
    es una pregunta sin respuesta: se redirige al humano."""
    seq = plan.turn_sequence
    for i, step in enumerate(seq):
        if step.action == "LOOK_AT_OTHER_ROBOT":
            siguiente = seq[i + 1] if i + 1 < len(seq) else None
            if not (siguiente and siguiente.speaker not in (step.speaker, "NONE")):
                print(f" ⚠️ [VALIDACIÓN] {step.speaker} miraba al otro robot sin respuesta; redirigido al humano.")
                step.action = "LOOK_AT_H1"
    return plan


async def execute_turn_plan(plan: TurnPlan, target: Optional[str] = None):
    try:
        if len(plan.turn_sequence) == 0:
            print("🤫 [ORQUESTADOR] Charla entre humanos o sin intervención. Robot en silencio.\n")
            await enviar_a_robots(target, {"type": "IDLE", "target": target})
            return

        for step in plan.turn_sequence:
            target_robot = step.speaker

            if target_robot in CONNECTED_ROBOTS:
                await broadcast_robot_speech_to_capturers(step.text)

                action_to_send = step.action
                if step.action == "LOOK_AT_OTHER_ROBOT":
                    action_to_send = "LOOK_AT_GABI" if target_robot == "ROBOT_DANI" else "LOOK_AT_DANI"

                payload = {
                    "type": "EXECUTE_TURN",
                    "text": step.text,
                    "emotion": step.emotion,
                    "action": action_to_send
                }
                print(f" -> [{target_robot}] Emotion: {step.emotion} | Action: {action_to_send} | Text: \"{step.text}\"")

                # Preparamos el evento ANTES de enviar para no perder un TURN_DONE rápido
                done_event = TURN_DONE_EVENTS.get(target_robot)
                if done_event:
                    done_event.clear()

                await CONNECTED_ROBOTS[target_robot].send(json.dumps(payload))

                # Esperamos TURN_DONE; la estimación por longitud es solo timeout de seguridad
                estimated = (len(step.text) / 14.0) + 0.8
                if done_event:
                    try:
                        await asyncio.wait_for(done_event.wait(), timeout=estimated + 4.0)
                    except asyncio.TimeoutError:
                        print(f" ⏱️ [{target_robot}] Sin TURN_DONE; continúo por timeout de seguridad.")
                else:
                    await asyncio.sleep(estimated)

                await asyncio.sleep(step.delay_after_ms / 1000.0)
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
                    TURN_DONE_EVENTS[client_id] = asyncio.Event()
                    print(f"[CONEXIÓN] Robot {client_id} conectado. Condición asignada: [{CURRENT_CONDITION}]")
                    await websocket.send(json.dumps({"status": "REGISTERED", "robot_id": client_id}))

                elif msg_type == "REGISTER_CAPTURER":
                    is_capturer = True
                    CONNECTED_CAPTURERS.add(websocket)
                    print("🔗 [CONEXIÓN] Capturador de audio registrado para filtro anti-eco.")

                elif msg_type == "TURN_DONE":
                    rid = data.get("robot_id") or client_id
                    ev = TURN_DONE_EVENTS.get(rid)
                    if ev:
                        ev.set()

                elif msg_type == "HUMAN_LISTENING_START":
                    target = data.get("target")
                    speaker = data.get("speaker", "Humano")
                    print(f"👂 [ESTADO] Escuchando a {speaker} (target: {target})...")
                    await enviar_a_robots(target, {"type": "LISTENING", "speaker": speaker, "target": target})

                elif msg_type == "HUMAN_LISTENING_TIMEOUT":
                    target = data.get("target")
                    await enviar_a_robots(target, {"type": "IDLE", "target": target})

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
                        await stop_all_robots(except_target=target)

                    await enviar_a_robots(target, {"type": "PROCESSING", "target": target})

                    plan = await get_turn_plan(CURRENT_CONDITION, speaker, text, target_robot=target)
                    plan = sanear_plan(plan)
                    CURRENT_EXECUTION_TASK = asyncio.create_task(execute_turn_plan(plan, target))

            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"⚠️ Error procesando mensaje: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if client_id and client_id in CONNECTED_ROBOTS:
            del CONNECTED_ROBOTS[client_id]
            TURN_DONE_EVENTS.pop(client_id, None)
            print(f"[DESCONEXIÓN] Robot {client_id} desconectado.")
        if is_capturer and websocket in CONNECTED_CAPTURERS:
            CONNECTED_CAPTURERS.discard(websocket)

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