import os
import mediapipe as mp

print("1 - MediaPipe:", mp.__version__)

modelo = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gesture_recognizer.task"
)

print("2 - Modelo:", modelo)
print("3 - Modelo existe:", os.path.exists(modelo))

if not os.path.exists(modelo):
    print("ERRO: gesture_recognizer.task nao existe.")
    raise SystemExit

try:
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

    print("4 - API carregada.")

    opcoes = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=modelo)
    )

    print("5 - Tentando carregar o Gesture Recognizer...")

    with GestureRecognizer.create_from_options(opcoes) as recognizer:
        print("6 - GESTURE RECOGNIZER FUNCIONANDO!")

except Exception as erro:
    print("ERRO AO CARREGAR:")
    print(type(erro).__name__)
    print(erro)