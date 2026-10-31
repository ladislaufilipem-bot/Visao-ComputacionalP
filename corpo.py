import cv2
import time
import mediapipe as mp


# ============================================================
# MEDIA PIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# ============================================================
# CONFIGURAÇÃO DO CORPO
# ============================================================

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CONFIGURAÇÃO DAS MÃOS
# ============================================================

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CONFIGURAÇÃO DO ROSTO
# ============================================================

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)


# ============================================================
# CÂMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("[ERRO] Não foi possível abrir a webcam.")
    exit()

print("==============================================")
print("       ECOBOT - VISAO HUMANA")
print("==============================================")
print("[ECOBOT] Corpo: ATIVADO")
print("[ECOBOT] Mãos: ATIVADAS")
print("[ECOBOT] Rosto: ATIVADO")
print("[ECOBOT] Webcam: ATIVADA")
print()
print("[ECOBOT] Coloque uma pessoa diante da câmera.")
print("[ECOBOT] ESC = sair")
print()


# ============================================================
# CONEXÕES DO CORPO
# ============================================================

POSE_CONNECTIONS = [
    # Tronco
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Braço esquerdo
    (11, 13),
    (13, 15),

    # Braço direito
    (12, 14),
    (14, 16),

    # Mão esquerda
    (15, 19),
    (15, 21),
    (15, 17),

    # Mão direita
    (16, 20),
    (16, 22),
    (16, 18),

    # Perna esquerda
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),

    # Perna direita
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),

    # Cabeça
    (7, 8),

    # Olhos / lateral da cabeça
    (2, 3),
    (4, 5),
    (1, 2),
    (1, 4),

    # Boca
    (9, 10)
]


# ============================================================
# CONEXÕES DAS MÃOS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# FUNÇÃO PARA DESENHAR PONTO
# ============================================================

def desenhar_ponto(frame, ponto, largura, altura, tamanho=4):

    x = int(ponto.x * largura)
    y = int(ponto.y * altura)

    cv2.circle(
        frame,
        (x, y),
        tamanho,
        (0, 255, 0),
        -1
    )

    return x, y


# ============================================================
# EXECUÇÃO
# ============================================================

with PoseLandmarker.create_from_options(pose_options) as pose, \
     HandLandmarker.create_from_options(hand_options) as hands, \
     FaceLandmarker.create_from_options(face_options) as face:

    while True:

        sucesso, frame = camera.read()

        if not sucesso:
            print("[ERRO] Falha ao capturar imagem.")
            break

        # Espelha a câmera
        frame = cv2.flip(frame, 1)

        altura, largura, _ = frame.shape

        # ----------------------------------------------------
        # CONVERSÃO
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagem = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp crescente em milissegundos
        timestamp = time.monotonic_ns() // 1_000_000

        # ----------------------------------------------------
        # DETECÇÃO DO CORPO
        # ----------------------------------------------------

        resultado_pose = pose.detect_for_video(
            imagem,
            timestamp
        )

        # ----------------------------------------------------
        # DETECÇÃO DAS MÃOS
        # ----------------------------------------------------

        resultado_maos = hands.detect_for_video(
            imagem,
            timestamp
        )

        # ----------------------------------------------------
        # DETECÇÃO DO ROSTO
        # ----------------------------------------------------

        resultado_rosto = face.detect_for_video(
            imagem,
            timestamp
        )

        # ====================================================
        # CORPO
        # ====================================================

        corpo_encontrado = False

        if resultado_pose.pose_landmarks:

            corpo_encontrado = True

            pontos_corpo = resultado_pose.pose_landmarks[0]

            # -----------------------------------------------
            # DESENHA PONTOS DO CORPO
            # -----------------------------------------------

            pontos_xy = []

            for ponto in pontos_corpo:

                x = int(ponto.x * largura)
                y = int(ponto.y * altura)

                pontos_xy.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # -----------------------------------------------
            # LIGA O ESQUELETO
            # -----------------------------------------------

            for a, b in POSE_CONNECTIONS:

                if a < len(pontos_xy) and b < len(pontos_xy):

                    cv2.line(
                        frame,
                        pontos_xy[a],
                        pontos_xy[b],
                        (255, 255, 255),
                        2
                    )

            # -----------------------------------------------
            # PONTOS IMPORTANTES
            # -----------------------------------------------

            nariz = pontos_corpo[0]

            ombro_esq = pontos_corpo[11]
            ombro_dir = pontos_corpo[12]

            quadril_esq = pontos_corpo[23]
            quadril_dir = pontos_corpo[24]

            # -----------------------------------------------
            # PESCOÇO
            # -----------------------------------------------

            pescoco_x = int(
                (
                    ombro_esq.x +
                    ombro_dir.x
                ) / 2 * largura
            )

            pescoco_y = int(
                (
                    ombro_esq.y +
                    ombro_dir.y
                ) / 2 * altura
            )

            nariz_x = int(nariz.x * largura)
            nariz_y = int(nariz.y * altura)

            # Desenha ligação cabeça -> pescoço

            cv2.line(
                frame,
                (nariz_x, nariz_y),
                (pescoco_x, pescoco_y),
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "PESCOCO",
                (pescoco_x + 5, pescoco_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # -----------------------------------------------
            # TRONCO
            # -----------------------------------------------

            torso_pts = [
                (
                    int(ombro_esq.x * largura),
                    int(ombro_esq.y * altura)
                ),
                (
                    int(ombro_dir.x * largura),
                    int(ombro_dir.y * altura)
                ),
                (
                    int(quadril_dir.x * largura),
                    int(quadril_dir.y * altura)
                ),
                (
                    int(quadril_esq.x * largura),
                    int(quadril_esq.y * altura)
                )
            ]

            for i in range(4):

                cv2.line(
                    frame,
                    torso_pts[i],
                    torso_pts[(i + 1) % 4],
                    (255, 0, 255),
                    3
                )

            cv2.putText(
                frame,
                "TRONCO",
                torso_pts[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

        # ====================================================
        # MÃOS
        # ====================================================

        quantidade_maos = len(
            resultado_maos.hand_landmarks
        )

        for mao in resultado_maos.hand_landmarks:

            pontos_mao = []

            for ponto in mao:

                x = int(ponto.x * largura)
                y = int(ponto.y * altura)

                pontos_mao.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

            for a, b in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    pontos_mao[a],
                    pontos_mao[b],
                    (0, 255, 255),
                    2
                )

        # ====================================================
        # ROSTO
        # ====================================================

        quantidade_rostos = len(
            resultado_rosto.face_landmarks
        )

        if quantidade_rostos > 0:

            rosto = resultado_rosto.face_landmarks[0]

            # Desenha os pontos faciais
            for ponto in rosto:

                x = int(ponto.x * largura)
                y = int(ponto.y * altura)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (255, 150, 0),
                    -1
                )

        # ====================================================
        # STATUS NA TELA
        # ====================================================

        if corpo_encontrado:
            status_corpo = "CORPO: OK"
        else:
            status_corpo = "CORPO: NAO DETECTADO"

        status = (
            f"{status_corpo} | "
            f"MAOS: {quantidade_maos} | "
            f"ROSTOS: {quantidade_rostos}"
        )

        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ====================================================
        # TERMINAL
        # ====================================================

        print(
            f"\r[ECOBOT] "
            f"Corpo={'SIM' if corpo_encontrado else 'NAO'} | "
            f"Mãos={quantidade_maos} | "
            f"Rostos={quantidade_rostos}",
            end=""
        )

        # ====================================================
        # MOSTRA IMAGEM
        # ====================================================

        cv2.imshow(
            "ECOBOT - Corpo Humano",
            frame
        )

        # ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break


camera.release()
cv2.destroyAllWindows()

print()
print("[ECOBOT] Sistema encerrado.")