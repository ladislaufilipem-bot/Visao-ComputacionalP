
import cv2
import time
import os
from collections import deque, Counter
import mediapipe as mp

# ==================================================
# CONFIGURACAO DO FACE LANDMARKER
# ==================================================

PASTA = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(PASTA, "face_landmarker.task")

if not os.path.isfile(MODELO):
    raise FileNotFoundError(
        "Nao encontrei face_landmarker.task na pasta do projeto."
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opcoes = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODELO),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)


# ==================================================
# LEITURA DOS BLENDSHAPES
# ==================================================

def media(valores, *nomes):
    return sum(valores.get(nome, 0.0) for nome in nomes) / len(nomes)


def classificar_expressao(b):
    sorriso = media(b, "mouthSmileLeft", "mouthSmileRight")
    boca_triste = media(b, "mouthFrownLeft", "mouthFrownRight")

    sobrancelha_baixa = media(
        b, "browDownLeft", "browDownRight"
    )
    sobrancelha_interna_alta = b.get("browInnerUp", 0.0)

    olhos_abertos = media(b, "eyeWideLeft", "eyeWideRight")
    olhos_semicerrados = media(b, "eyeSquintLeft", "eyeSquintRight")

    boca_aberta = b.get("jawOpen", 0.0)

    boca_estendida = media(
        b, "mouthStretchLeft", "mouthStretchRight"
    )
    nariz_contraido = media(
        b, "noseSneerLeft", "noseSneerRight"
    )

    # FELICIDADE: sorriso evidente
    if sorriso >= 0.40 and sorriso > boca_triste:
        return "FELIZ"

    # RAIVA: sobrancelhas baixas e outros sinais associados
    if (
        sobrancelha_baixa >= 0.40
        and (
            olhos_semicerrados >= 0.18
            or nariz_contraido >= 0.18
            or boca_triste >= 0.20
        )
    ):
        return "RAIVA"

    # MEDO: olhos arregalados, sobrancelhas elevadas
    # e boca estendida. Regra heuristica inicial.
    if (
        olhos_abertos >= 0.35
        and sobrancelha_interna_alta >= 0.25
        and boca_estendida >= 0.20
    ):
        return "MEDO"

    # SURPRESA: olhos arregalados e boca aberta
    if (
        olhos_abertos >= 0.40
        and sobrancelha_interna_alta >= 0.25
        and boca_aberta >= 0.30
    ):
        return "SURPRESA"

    # TRISTEZA: cantos da boca voltados para baixo
    if (
        boca_triste >= 0.30
        and sorriso < 0.25
    ):
        return "TRISTE"

    return "NEUTRA / INDEFINIDA"


# ==================================================
# CAMERA
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam.")

historico = deque(maxlen=7)
expressao_atual = "PROCURANDO ROSTO"
ultima_mensagem = None

print("========================================")
print("     ECOBOT - EXPRESSOES FACIAIS")
print("========================================")
print("Expressoes: feliz, triste, raiva, medo,")
print("surpresa e neutra/indefinida.")
print("Pressione ESC para sair.")
print()


# ==================================================
# PROCESSAMENTO DO VIDEO
# ==================================================

try:
    with FaceLandmarker.create_from_options(opcoes) as detector:

        ultimo_timestamp = 0

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
            timestamp = max(timestamp, ultimo_timestamp + 1)
            ultimo_timestamp = timestamp

            resultado = detector.detect_for_video(
                imagem, timestamp
            )

            if (
                resultado.face_landmarks
                and resultado.face_blendshapes
            ):
                # Converte os coeficientes em um dicionario
                categorias = resultado.face_blendshapes[0]

                valores = {
                    item.category_name: item.score
                    for item in categorias
                }

                previsao = classificar_expressao(valores)
                historico.append(previsao)

                # Suaviza os resultados usando varios frames
                frequencias = Counter(historico)
                candidata, quantidade = frequencias.most_common(1)[0]

                if quantidade >= 4:
                    expressao_atual = candidata

                # Desenha um retangulo ao redor do rosto
                pontos = resultado.face_landmarks[0]

                xs = [int(p.x * largura) for p in pontos]
                ys = [int(p.y * altura) for p in pontos]

                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(largura - 1, max(xs))
                y2 = min(altura - 1, max(ys))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            else:
                historico.clear()
                expressao_atual = "ROSTO NAO DETECTADO"

            # Mostra a expressao na imagem
            cv2.putText(
                frame,
                "Expressao: " + expressao_atual,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Terminal: imprime apenas quando muda
            if expressao_atual != ultima_mensagem:
                print(
                    f"[ECOBOT] Expressao facial estimada: "
                    f"{expressao_atual}"
                )
                ultima_mensagem = expressao_atual

            cv2.imshow("ECOBOT - Expressoes Faciais", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("[ECOBOT] Sistema encerrado.")
