
import cv2
import time
import os

import mediapipe as mp

from collections import defaultdict, deque
from statistics import median, mean


# ==========================================================
# 1. CONFIGURACOES GERAIS
# ==========================================================

PASTA = os.path.dirname(os.path.abspath(__file__))

MODO_MODELO = "precisao"

MODELOS = {
    "precisao": "efficientdet_lite2.tflite",
    "rapido": "efficientdet_lite0.tflite"
}

ARQUIVO_MODELO = MODELOS[MODO_MODELO]
MODELO = os.path.join(PASTA, ARQUIVO_MODELO)

# Configuracoes da camera
LARGURA_CAMERA = 640
ALTURA_CAMERA = 480
FPS_DESEJADO = 30

# Detector
MAX_RESULTADOS = 8

# O detector aceita candidatos a partir deste valor.
LIMIAR_INTERNO = 0.30

# Nosso filtro adicional de confianca.
LIMIAR_GERAL = 0.65

# Categorias que exigem confianca maior.
LIMIARES_ESPECIAIS = {
    "toothbrush": 0.85
}

# Estabilidade temporal.
TAMANHO_HISTORICO = 5
FRAMES_PARA_CONFIRMAR = 3


# ==========================================================
# 2. VERIFICACAO DO MODELO
# ==========================================================

if not os.path.isfile(MODELO):

    if MODO_MODELO == "precisao":
        raise FileNotFoundError(
            "Modelo Lite2 nao encontrado: "
            + MODELO
            + "\nBaixe o modelo ou altere MODO_MODELO para 'rapido'."
        )

    raise FileNotFoundError(
        "Modelo Lite0 nao encontrado: " + MODELO
    )

print("=" * 55)
print("          ECOBOT - VISAO DE OBJETOS")
print("=" * 55)

print("[ECOBOT] Modelo:", ARQUIVO_MODELO)
print("[ECOBOT] Modo:", MODO_MODELO)


# ==========================================================
# 3. TRADUCAO DAS CATEGORIAS
# ==========================================================

TRADUCAO = {
    "person": "pessoa",
    "bicycle": "bicicleta",
    "car": "carro",
    "motorcycle": "motocicleta",
    "airplane": "aviao",
    "bus": "onibus",
    "train": "trem",
    "truck": "caminhao",
    "boat": "barco",
    "traffic light": "semaforo",
    "fire hydrant": "hidrante",
    "stop sign": "placa de pare",
    "parking meter": "parquimetro",
    "bench": "banco",
    "bird": "passaro",
    "cat": "gato",
    "dog": "cachorro",
    "horse": "cavalo",
    "sheep": "ovelha",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "urso",
    "zebra": "zebra",
    "giraffe": "girafa",
    "backpack": "mochila",
    "umbrella": "guarda-chuva",
    "handbag": "bolsa",
    "tie": "gravata",
    "suitcase": "mala",
    "frisbee": "frisbee",
    "skis": "esquis",
    "snowboard": "snowboard",
    "sports ball": "bola",
    "kite": "pipa",
    "baseball bat": "taco",
    "baseball glove": "luva",
    "skateboard": "skate",
    "surfboard": "prancha",
    "tennis racket": "raquete",
    "bottle": "garrafa",
    "wine glass": "copo de vinho",
    "cup": "copo",
    "fork": "garfo",
    "knife": "faca",
    "spoon": "colher",
    "bowl": "tigela",
    "banana": "banana",
    "apple": "maca",
    "sandwich": "sanduiche",
    "orange": "laranja",
    "broccoli": "brocolis",
    "carrot": "cenoura",
    "hot dog": "cachorro-quente",
    "pizza": "pizza",
    "donut": "rosquinha",
    "cake": "bolo",
    "chair": "cadeira",
    "couch": "sofa",
    "potted plant": "planta",
    "bed": "cama",
    "dining table": "mesa",
    "toilet": "vaso sanitario",
    "tv": "televisao",
    "laptop": "notebook",
    "mouse": "mouse",
    "remote": "controle remoto",
    "keyboard": "teclado",
    "cell phone": "celular",
    "microwave": "micro-ondas",
    "oven": "forno",
    "toaster": "torradeira",
    "sink": "pia",
    "refrigerator": "geladeira",
    "book": "livro",
    "clock": "relogio",
    "vase": "vaso",
    "scissors": "tesoura",
    "teddy bear": "urso de pelucia",
    "hair drier": "secador",
    "toothbrush": "escova de dentes"
}


# ==========================================================
# 4. MEDIA PIPE OBJECT DETECTOR
# ==========================================================

BaseOptions = mp.tasks.BaseOptions

ObjectDetector = mp.tasks.vision.ObjectDetector

ObjectDetectorOptions = (
    mp.tasks.vision.ObjectDetectorOptions
)

RunningMode = mp.tasks.vision.RunningMode


opcoes = ObjectDetectorOptions(

    base_options=BaseOptions(
        model_asset_path=MODELO
    ),

    running_mode=RunningMode.VIDEO,

    max_results=MAX_RESULTADOS,

    score_threshold=LIMIAR_INTERNO
)


# ==========================================================
# 5. CAMERA
# ==========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam.")

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    LARGURA_CAMERA
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    ALTURA_CAMERA
)

camera.set(
    cv2.CAP_PROP_FPS,
    FPS_DESEJADO
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

print("[ECOBOT] Resolucao:", LARGURA_CAMERA, "x", ALTURA_CAMERA)
print("[ECOBOT] Detector configurado.")
print("[ECOBOT] Inicializando...")
print()


# ==========================================================
# 6. HISTORICO DE DETECCOES
# ==========================================================

historico = deque(maxlen=TAMANHO_HISTORICO)

ultimo_timestamp = 0

ultimo_relatorio = time.perf_counter()

fps_inicio = time.perf_counter()
fps_frames = 0

fps_atual = 0.0


# ==========================================================
# 7. FUNCAO DE CONFIRMACAO
# ==========================================================

def confirmar_objetos(historico):

    nomes = set()

    for quadro in historico:
        nomes.update(quadro.keys())

    confirmados = {}

    for nome in nomes:

        ocorrencias = [
            quadro[nome]
            for quadro in historico
            if nome in quadro
        ]

        if len(ocorrencias) < FRAMES_PARA_CONFIRMAR:
            continue

        quantidades = [
            item["quantidade"]
            for item in ocorrencias
        ]

        confiancas = [
            item["confianca"]
            for item in ocorrencias
        ]

        confirmados[nome] = {
            "quantidade": max(
                1,
                int(round(median(quantidades)))
            ),

            "confianca": mean(confiancas)
        }

    return confirmados


# ==========================================================
# 8. EXECUCAO
# ==========================================================

try:

    with ObjectDetector.create_from_options(opcoes) as detector:

        print("[ECOBOT] Detector carregado!")
        print("[ECOBOT] Analisando ambiente...")
        print("[ECOBOT] ESC para encerrar.")
        print()

        while True:

            inicio_frame = time.perf_counter()

            sucesso, frame = camera.read()

            if not sucesso:
                print("[ERRO] Falha na captura da webcam.")
                break

            frame = cv2.flip(frame, 1)

            altura, largura = frame.shape[:2]

            # ------------------------------------------
            # CONVERSAO DA IMAGEM
            # ------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            imagem = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Timestamp crescente
            timestamp = time.monotonic_ns() // 1_000_000

            timestamp = max(
                timestamp,
                ultimo_timestamp + 1
            )

            ultimo_timestamp = timestamp

            # ------------------------------------------
            # DETECCAO
            # ------------------------------------------

            resultado = detector.detect_for_video(
                imagem,
                timestamp
            )

            objetos_quadro = defaultdict(list)

            deteccoes_visuais = []

            # ------------------------------------------
            # PROCESSAMENTO
            # ------------------------------------------

            for deteccao in resultado.detections:

                if not deteccao.categories:
                    continue

                categoria = max(
                    deteccao.categories,
                    key=lambda item: item.score
                )

                original = (
                    categoria.category_name
                    or categoria.display_name
                    or "objeto"
                ).lower()

                confianca = float(categoria.score)

                limiar = max(
                    LIMIAR_GERAL,
                    LIMIARES_ESPECIAIS.get(original, 0.0)
                )

                # Descarta classificacoes pouco confiaveis
                if confianca < limiar:
                    continue

                nome = TRADUCAO.get(
                    original,
                    original
                )

                caixa = deteccao.bounding_box

                x1 = max(0, caixa.origin_x)
                y1 = max(0, caixa.origin_y)

                x2 = min(
                    largura - 1,
                    caixa.origin_x + caixa.width
                )

                y2 = min(
                    altura - 1,
                    caixa.origin_y + caixa.height
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # Evita caixas muito pequenas
                area = (x2 - x1) * (y2 - y1)
                area_relativa = area / (largura * altura)

                if area_relativa < 0.001:
                    continue

                objetos_quadro[nome].append(confianca)

                deteccoes_visuais.append({
                    "nome": nome,
                    "confianca": confianca,
                    "caixa": (x1, y1, x2, y2)
                })

            # ------------------------------------------
            # RESUMO DO FRAME
            # ------------------------------------------

            resumo_quadro = {}

            for nome, confiancas in objetos_quadro.items():

                resumo_quadro[nome] = {
                    "quantidade": len(confiancas),
                    "confianca": mean(confiancas)
                }

            historico.append(resumo_quadro)

            # Confirma somente classes estaveis
            objetos_confirmados = confirmar_objetos(
                historico
            )

            # ------------------------------------------
            # DESENHO DAS DETECCOES
            # ------------------------------------------

            for item in deteccoes_visuais:

                nome = item["nome"]
                confianca = item["confianca"]

                x1, y1, x2, y2 = item["caixa"]

                confirmado = nome in objetos_confirmados

                if confirmado:
                    cor = (0, 255, 0)
                    identificacao = nome
                else:
                    cor = (0, 200, 255)
                    identificacao = "verificando: " + nome

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    cor,
                    2
                )

                texto = (
                    f"{identificacao} "
                    f"{confianca * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    texto,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cor,
                    2
                )

            # ------------------------------------------
            # FPS
            # ------------------------------------------

            fps_frames += 1

            agora = time.perf_counter()

            if agora - fps_inicio >= 1.0:

                fps_atual = fps_frames / (
                    agora - fps_inicio
                )

                fps_frames = 0
                fps_inicio = agora

            # ------------------------------------------
            # TERMINAL
            # ------------------------------------------

            if agora - ultimo_relatorio >= 1.0:

                if objetos_confirmados:

                    partes = []

                    for nome, dados in sorted(
                        objetos_confirmados.items()
                    ):

                        partes.append(
                            f"{nome}: {dados['quantidade']} "
                            f"({dados['confianca'] * 100:.1f}%)"
                        )

                    print(
                        "[ECOBOT] Objetos confirmados: "
                        + " | ".join(partes),
                        flush=True
                    )

                elif deteccoes_visuais:

                    print(
                        "[ECOBOT] Verificando classificacoes...",
                        flush=True
                    )

                else:

                    print(
                        "[ECOBOT] Nenhum objeto reconhecido "
                        "com confianca suficiente.",
                        flush=True
                    )

                print(
                    f"[ECOBOT] FPS: {fps_atual:.1f}",
                    flush=True
                )

                ultimo_relatorio = agora

            # ------------------------------------------
            # INFORMACOES NA TELA
            # ------------------------------------------

            total = len(deteccoes_visuais)

            cv2.putText(
                frame,
                f"OBJETOS: {total}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps_atual:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "MODELO: " + MODO_MODELO.upper(),
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ECOBOT - Reconhecimento de Objetos",
                frame
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

print()
print("[ECOBOT] Sistema encerrado.")
