
import cv2
import time
import math
import os
import mediapipe as mp

# ==================================================
# MODELO E CONFIGURACAO
# ==================================================

PASTA = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(PASTA, "gesture_recognizer.task")

if not os.path.isfile(MODELO):
    raise FileNotFoundError(
        "Nao encontrei gesture_recognizer.task na pasta do projeto."
    )

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

opcoes = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODELO),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================================================
# CLASSIFICACAO PERSONALIZADA
# ==================================================

def distancia(a, b, largura, altura):
    dx = (a.x - b.x) * largura
    dy = (a.y - b.y) * altura
    return math.hypot(dx, dy)


def dedo_levantado(pontos, ponta, articulacao):
    # A imagem tem origem no canto superior esquerdo.
    # Esta regra funciona melhor com a mao na vertical.
    return pontos[ponta].y < pontos[articulacao].y - 0.02


def gesto_personalizado(pontos, largura, altura):
    indicador = dedo_levantado(pontos, 8, 6)
    medio = dedo_levantado(pontos, 12, 10)
    anelar = dedo_levantado(pontos, 16, 14)
    mindinho = dedo_levantado(pontos, 20, 18)

    largura_palma = distancia(
        pontos[5], pontos[17], largura, altura
    )

    distancia_polegar = distancia(
        pontos[4], pontos[5], largura, altura
    )

    polegar_estendido = (
        largura_palma > 0 and
        distancia_polegar > largura_palma * 0.35
    )

    # ROCK: indicador e mindinho levantados,
    # medio e anelar dobrados e polegar estendido.
    if (
        indicador and mindinho and
        not medio and not anelar and
        polegar_estendido
    ):
        return ("ROCK 🤘", "ROCK")

    # DEDO DO MEIO: somente o dedo medio levantado
    # entre os quatro dedos principais.
    if (
        medio and
        not indicador and
        not anelar and
        not mindinho
    ):
        return ("DEDO DO MEIO 🖕", "DEDO DO MEIO")

    return None


# ==================================================
# GESTOS NATIVOS DO MEDIAPIPE
# ==================================================

GESTOS_NATIVOS = {
    "Thumb_Up": ("JOINHA 👍", "JOINHA"),
    "Open_Palm": ("MAO ABERTA ✋", "MAO ABERTA"),
    "Closed_Fist": ("PUNHO FECHADO ✊", "PUNHO FECHADO")
}

# Conexoes da mao para desenhar na tela
CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ==================================================
# CAMERA
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Nao consegui abrir a webcam.")

print("======================================")
print("      ECOBOT - RECONHECIMENTO")
print("======================================")
print("Gestos: joinha, mao aberta, punho,")
print("dedo do meio e rock.")
print("Mostre um gesto diante da camera.")
print("ESC = sair")
print()

estado_confirmado = None
candidato = None
contador_candidato = 0

# ==================================================
# LOOP PRINCIPAL
# ==================================================

try:
    with GestureRecognizer.create_from_options(opcoes) as reconhecedor:
        while True:
            sucesso, frame = camera.read()

            if not sucesso:
                print("[ERRO] Falha ao capturar imagem.")
                break

            frame = cv2.flip(frame, 1)
            altura, largura = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagem = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = time.monotonic_ns() // 1_000_000

            resultado = reconhecedor.recognize_for_video(
                imagem, timestamp
            )

            gesto_detectado = None

            # Processamos uma mao, conforme num_hands=1.
            if resultado.hand_landmarks:
                pontos = resultado.hand_landmarks[0]

                # Primeiro tentamos os gestos personalizados.
                gesto_detectado = gesto_personalizado(
                    pontos, largura, altura
                )

                # Depois usamos o classificador nativo.
                if gesto_detectado is None and resultado.gestures:
                    categorias = resultado.gestures[0]

                    if categorias:
                        categoria = categorias[0]

                        if categoria.score >= 0.60:
                            gesto_detectado = GESTOS_NATIVOS.get(
                                categoria.category_name
                            )

                # Desenha os pontos e as conexoes da mao.
                xy = []
                for ponto in pontos:
                    px = int(ponto.x * largura)
                    py = int(ponto.y * altura)
                    xy.append((px, py))
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                for a, b in CONEXOES:
                    cv2.line(
                        frame, xy[a], xy[b], (0, 255, 255), 2
                    )

            # Exige 3 frames consecutivos para reduzir oscilacoes.
            if gesto_detectado == candidato:
                contador_candidato += 1
            else:
                candidato = gesto_detectado
                contador_candidato = 1

            if (
                contador_candidato >= 3
                and candidato != estado_confirmado
            ):
                estado_confirmado = candidato

                if estado_confirmado is None:
                    print("[ECOBOT] Gesto liberado.")
                else:
                    print(f"[ECOBOT] {estado_confirmado[0]}")

            # Texto na janela da camera.
            if estado_confirmado is None:
                texto = "Mostre um gesto"
            else:
                texto = estado_confirmado[1]

            cv2.putText(
                frame,
                texto,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("ECOBOT - Gestos", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("[ECOBOT] Programa encerrado.")
