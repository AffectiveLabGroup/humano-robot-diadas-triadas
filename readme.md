# Documentación del Sistema: Orquestador y Capturador de Diálogo Robot-Humano

Este proyecto es una plataforma de **interacción humano-robot en tiempo real** que coordina la conversación natural entre humanos y robots sociales (**Sanbot - ROBOT_ALEX y ROBOT_ROBIN**).

El sistema gestiona la captura de voz, la identificación del hablante por huella vocal, el filtrado de ruido/anti-eco y la generación de respuestas de diálogo contextuales y gestuales mediante Inteligencia Artificial.

---

## Arquitectura General

El ecosistema se compone de **tres bloques principales**:

```
 ┌──────────────────────────┐           WebSocket           ┌──────────────────────────────┐
 │   Capturador de Voz      │ ────────────────────────────> │     Servidor Orquestador     │
 │ (Reconocimiento + Biometría) │ <──────────────────────────── │ (OpenAI + Control de Turnos) │
 └──────────────────────────┘      Notificación Anti-Eco     └──────────────┬───────────────┘
                                                                           │ WebSocket
                                                                           ▼
                                                             ┌──────────────────────────────┐
                                                             │      Robots Sociales         │
                                                             │  (ROBOT_ALEX / ROBOT_ROBIN)  │
                                                             └──────────────────────────────┘

```

1. **Capturador de Audio (`capturador_rec_wake.py`)**: Escucha el ambiente, detecta si se ha dicho una palabra de activación (Wake Word), identifica quién está hablando mediante su huella vocal y envía el texto transcrito al orquestador.
2. **Servidor Orquestador (`server.py o server_optimizado.py`)**: Mantiene el estado global, el historial de la conversación y utiliza la API de OpenAI (`gpt-4o-mini`) para decidir qué robot habla, qué dice, qué emoción muestra en pantalla y hacia dónde mira.
3. **Robots Sanbot (Clientes WS)**: Reciben las órdenes de ejecución en tiempo real (`EXECUTE_TURN`) y reproducen el texto por altavoz mientras: se cambia su pantalla para indicar el turno de habla, mueven su estructura (movimiento de cabeza hacia el hablante) y cambian su expresión facial (según emoción que surge de la conversación).

---

## 1. Módulo Capturador de Voz (`capturador_rec_wake.py`)

### ¿Qué hace?

Es la "oreja" del sistema. Escucha en bucle el micrófono ambiental sin bloquear el resto de procesos, filtra el ruido, busca la palabra de activación (*Wake Word*) e identifica la voz de la persona.

### Componentes Clave:

* **Audio mediante `sounddevice`**: Graba trozos de audio en bucle de forma nativa.
* **Motor Biométrico (`Resemblyzer`)**: Carga en memoria una base de datos con las huellas vocales de los usuarios conocidos (`voices/`). Al capturar un audio, calcula la distancia euclidiana entre la voz actual y la base de datos para identificar el nombre de la persona (ej. *Paula, Loreto, Liany, Juan Jesús*) o etiquetarlo como *"Desconocido"*.
* **Normalizador y Wake Word**: Convierte el texto a minúsculas, elimina acentos/tildes y valida si el texto contiene palabras clave como `"alex"`, `"alexa"`, `"robin"`, etc.
* **Canal Anti-Eco WebSocket**: Se conecta al orquestador para recibir los mensajes `ROBOT_SPOKE`. Esto almacena las frases recientes del robot en una memoria temporal para evitar que el micrófono se auto-escuche y entre en un bucle infinito (se incluye en la versión de reconocimiento contínuo).

### Flujo de Trabajo del Capturador:

1. Graba bloques de audio de 4 segundos a 16 kHz.
2. Transcribe el audio usando **Google Speech API** (`speech_recognition`).
3. Comprueba si el texto contiene una **Wake Word**. Si no la contiene, lo descarta (`💤 IGNORADO`).
4. Si la contiene, procesa el vector de voz con **Resemblyzer** y determina el nombre del hablante.
5. Envía un paquete JSON tipo `HUMAN_INPUT` al servidor WebSocket.

---

## 2. Servidor Orquestador (`server.py`)

### ¿Qué hace?

Administra la lógica de la conversación, controla las interrupciones en tiempo real y calcula la mejor respuesta robótica mediante el modelo de lenguaje gpt-mini.

### Componentes Clave:

#### A. Gestión de Condición Experimental (`CURRENT_CONDITION`)

El servidor ajusta automáticamente el comportamiento del diálogo en función del escenario de pruebas activo:

* **Condición A (1H + 1R)**: 1 Humano + `ROBOT_ALEX`.
* **Condición B (2H + 1R)**: 2 Humanos + `ROBOT_ALEX`. Si los humanos hablan entre sí, el robot se mantiene en silencio.
* **Condición C (1H + 2R)**: 1 Humano + `ROBOT_ALEX` + `ROBOT_ROBIN`.
* **Condición D (2H + 2R)**: 2 Humanos + `ROBOT_ALEX` + `ROBOT_ROBIN`.

#### B. Generación de Turnos Optimizada (`get_turn_plan`)

Para minimizar la latencia y lograr que los robots respondan lo más rápido posible:

* **Filtro de Prompt Dinámico**: Solo envía a OpenAI las instrucciones pertinentes para la condición activa (server_optimizado.py reduce los tokens de entrada de ~1200 a ~400).
* **Historial Acotado**: Envía únicamente los últimos 4 mensajes relevantes de la conversación.
* **Estructura Rígida (Pydantic / Structured Outputs)**: Exige a la API que devuelva un objeto con la estructura `TurnPlan`:
```json
{
  "condition": "C",
  "addressed_to": "H1",
  "turn_sequence": [
    {
      "speaker": "ROBOT_ALEX",
      "text": "Yo prefiero Madrid, los sueldos de dos mil cien euros compensan el alquiler.",
      "emotion": "HAPPY",
      "action": "LOOK_AT_H1",
      "delay_after_ms": 500
    }
  ],
  "requires_human_input": true
}

```



#### C. Control de Interrupciones en Tiempo Real

Si un robot está ejecutando un turno de voz y el capturador detecta una nueva frase humana (`HUMAN_INPUT`):

1. El orquestador cancela inmediatamente la tarea asíncrona de ejecución (`CURRENT_EXECUTION_TASK.cancel()`).
2. Envía un mensaje de emergencia `STOP_SPEECH` a todos los robots para cortar el audio al instante.
3. Procesa el nuevo mensaje del humano.

---

## Protocolo de Comunicación (WebSockets - Puerto 8765)

| Emisor | Receptor | Tipo de Mensaje (`type`) | Descripción |
| --- | --- | --- | --- |
| **Robot** | Servidor | `REGISTER` | Registra al Sanbot enviando su `robot_id` y la `condition`. |
| **Capturador** | Servidor | `REGISTER_CAPTURER` | Suscribe al script de micro al canal Anti-Eco. |
| **Capturador** | Servidor | `HUMAN_INPUT` | Envía el texto transcrito y el nombre del hablante identificado. |
| **Servidor** | Robot | `EXECUTE_TURN` | Ordena al robot decir una frase, mostrar una emoción y realizar un gesto. |
| **Servidor** | Robot | `STOP_SPEECH` | Orden de cancelación inmediata por interrupción humana. |
| **Servidor** | Capturador | `ROBOT_SPOKE` | Notifica al capturador el texto que va a pronunciar el robot para evitar ecos. |

---

## Guía de Inicio Rápido

### Requisitos Previos

* **Python 3.9 - 3.11**
* Clave de API de OpenAI configurada en la variable de entorno `OPENAI_API_KEY`.
* Muestras de audio de referencia en la carpeta `voices/` (`.wav` a 16kHz) para la identificación de voz.

### Instalar Dependencias

```bash
pip install asyncio websockets pydantic openai speechrecognition sounddevice numpy soundfile scipy resemblyzer

```

### Ejecución del Sistema

1. **Iniciar el Servidor Orquestador**:
```bash
python server.py

```


*(El servidor se iniciará en `ws://localhost:8765`)*.
2. **Iniciar el Capturador de Micrófono**:
```bash
python capturador_rec_wake.py

```


3. **Conectar los Robots Sanbot**:
Conectar las aplicaciones cliente de los robots apuntando a la IP del Servidor Orquestador.

