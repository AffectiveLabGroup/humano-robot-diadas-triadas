"""
Graba muestras de voz en WAV (16 kHz, mono, 16 bit) para el capturador (Resemblyzer).

Instalación (una vez):
    pip install sounddevice soundfile

Uso:
    python grabar_voz.py

Los archivos se guardan en ./voices con el mismo patrón que usa el capturador:
    paula_ref.wav, paula_ref2.wav, paula_ref3.wav ...
"""
import os

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CARPETA = "voices"
DURACION_POR_DEFECTO = 15  # segundos


def siguiente_ruta(nombre: str) -> str:
    """Primer nombre libre: nombre_ref.wav, nombre_ref2.wav, nombre_ref3.wav..."""
    n = 1
    while True:
        archivo = f"{nombre}_ref.wav" if n == 1 else f"{nombre}_ref{n}.wav"
        ruta = os.path.join(CARPETA, archivo)
        if not os.path.exists(ruta):
            return ruta
        n += 1


def pedir_entero(mensaje: str, defecto):
    valor = input(mensaje).strip()
    if not valor:
        return defecto
    try:
        return int(valor)
    except ValueError:
        print("Valor no válido, uso el de por defecto.")
        return defecto


def main():
    os.makedirs(CARPETA, exist_ok=True)

    print("\nDispositivos de audio disponibles:\n")
    print(sd.query_devices())
    device = pedir_entero("\nID del micrófono (Enter = el de por defecto de Windows): ", None)

    nombre = input("Nombre de la persona (ej. paula): ").strip().lower().replace(" ", "")
    if not nombre:
        print("Hace falta un nombre.")
        return

    segundos = pedir_entero(f"Duración de cada muestra en segundos (Enter = {DURACION_POR_DEFECTO}): ",
                            DURACION_POR_DEFECTO)

    print("\nConsejo: habla con naturalidad (cuenta qué hiciste ayer, opina sobre dónde te gustaría vivir...).")
    print("Graba 3-4 muestras en el mismo sitio y con el mismo micro que usarás en el experimento.\n")

    while True:
        accion = input("Enter = grabar nueva muestra | q = salir: ").strip().lower()
        if accion == "q":
            break

        ruta = siguiente_ruta(nombre)
        print(f"🎙️  Grabando {segundos} s → habla ahora...")
        audio = sd.rec(int(segundos * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="int16", device=device)
        sd.wait()
        print("⏹️  Grabación terminada. Reproduciendo para que la compruebes...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

        if input("¿Guardar esta muestra? (s/n): ").strip().lower() == "s":
            sf.write(ruta, audio, SAMPLE_RATE, subtype="PCM_16")
            print(f"✅ Guardado: {ruta}\n")
        else:
            print("Descartada.\n")

    print("\nRecuerda añadir las rutas nuevas a 'voces_config' en el capturador.")


if __name__ == "__main__":
    main()