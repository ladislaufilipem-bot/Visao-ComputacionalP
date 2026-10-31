import cv2
import time
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: nao foi possivel abrir a webcam.")
    exit()

ultimo_timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        sucesso, frame = camera.read()

        if not sucesso:
            print("Erro ao capturar imagem.")
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        imagem = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp = int(time.time() * 1000)

        if timestamp <= ultimo_timestamp:
            timestamp = ultimo_timestamp + 1

        ultimo_timestamp = timestamp

        resultado = landmarker.detect_for_video(
            imagem,
            timestamp
        )

        altura, largura, _ = frame.shape

        if resultado.hand_landmarks:

            for mao in resultado.hand_landmarks:

                pontos = []

                # Os 21 pontos da mão
                for ponto in mao:
                    x = int(ponto.x * largura)
                    y = int(ponto.y * altura)

                    pontos.append((x, y))

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Ligações dos dedos
                conexoes = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17)
                ]

                for a, b in conexoes:
                    cv2.line(
                        frame,
                        pontos[a],
                        pontos[b],
                        (255, 255, 255),
                        2
                    )

        cv2.imshow("ECOBOT - Scanner de Maos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

camera.release()
cv2.destroyAllWindows()