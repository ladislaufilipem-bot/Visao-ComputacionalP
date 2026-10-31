import cv2
import time
import mediapipe as mp

# --------------------------------------------------
# MEDIA PIPE
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

RunningMode = mp.tasks.vision.RunningMode


# --------------------------------------------------
# CONFIGURAÇÃO
# --------------------------------------------------

opcoes = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path="gesture_recognizer.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# CÂMERA
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("[ERRO] Não consegui abrir a webcam.")
    exit()

print("====================================")
print("       ECOBOT - DETECTOR DE JOINHA")
print("====================================")
print("Faça 👍 para testar.")
print("ESC = sair")
print()


# --------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------

ultimo_joinha = False
timestamp = 0

with GestureRecognizer.create_from_options(opcoes) as recognizer:

    while True:

        sucesso, frame = camera.read()

        if not sucesso:
            print("[ERRO] Não consegui capturar a imagem.")
            break

        frame = cv2.flip(frame, 1)

        # Converte BGR -> RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagem = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp crescente
        timestamp = time.monotonic_ns() // 1_000_000

        resultado = recognizer.recognize_for_video(
            imagem,
            timestamp
        )

        joinha = False

        # Verifica os gestos detectados
        for mao in resultado.gestures:

            if len(mao) > 0:

                gesto = mao[0]

                if (
                    gesto.category_name == "Thumb_Up"
                    and gesto.score >= 0.65
                ):
                    joinha = True

        # --------------------------------------------------
        # TERMINAL
        # --------------------------------------------------

        if joinha and not ultimo_joinha:
            print("[ECOBOT] JOINHA 👍")

        if not joinha and ultimo_joinha:
            print("[ECOBOT] Joinha saiu.")

        ultimo_joinha = joinha

        # --------------------------------------------------
        # CÂMERA
        # --------------------------------------------------

        cv2.putText(
            frame,
            "JOINHA!" if joinha else "Mostrando camera",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "ECOBOT - Joinha",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break


camera.release()
cv2.destroyAllWindows()

print("[ECOBOT] Encerrado.")