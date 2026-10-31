import cv2
import os
import time

print("======================================")
print("       ECOBOT - VISAO COMPUTACIONAL")
print("======================================")
print()

# Local do arquivo do detector
arquivo = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml"
)

# Carrega o detector
cascata = cv2.CascadeClassifier(arquivo)

if cascata.empty():
    print("[ERRO] Nao consegui carregar o detector de rostos.")
    exit()

print("[ECOBOT] Detector de rostos carregado.")

# Abre a webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("[ERRO] Nao consegui abrir a webcam.")
    exit()

print("[ECOBOT] Webcam funcionando.")
print("[ECOBOT] Iniciando visao...")
print()

ultimo_aviso = 0

while True:

    sucesso, frame = camera.read()

    if not sucesso:
        print("[ERRO] Falha ao capturar imagem.")
        break

    # Espelha a imagem
    frame = cv2.flip(frame, 1)

    # Escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos
    rostos = cascata.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    quantidade = len(rostos)

    # Descobre o que o ECOBOT está vendo
    if quantidade == 0:

        texto = "NENHUMA PESSOA"

    elif quantidade == 1:

        x, y, largura, altura = rostos[0]

        # Centro do rosto
        centro_x = x + largura / 2

        # Centro da tela
        centro_tela = frame.shape[1] / 2

        # Posição horizontal
        if centro_x < centro_tela - 100:
            posicao = "A ESQUERDA"

        elif centro_x > centro_tela + 100:
            posicao = "A DIREITA"

        else:
            posicao = "NO CENTRO"

        # Distância aproximada pela largura do rosto
        if largura < 100:
            distancia = "LONGE"

        elif largura < 200:
            distancia = "DISTANCIA MEDIA"

        else:
            distancia = "PERTO"

        texto = f"1 PESSOA | {posicao} | {distancia}"

    else:

        texto = f"{quantidade} PESSOAS"

    # Mostra no terminal aproximadamente 1 vez por segundo
    agora = time.time()

    if agora - ultimo_aviso >= 1:
        print(f"[ECOBOT] Estou vendo: {texto}")
        ultimo_aviso = agora

    # Desenha os rostos
    for x, y, largura, altura in rostos:

        cv2.rectangle(
            frame,
            (x, y),
            (x + largura, y + altura),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "ROSTO",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Mostra a câmera
    cv2.imshow("ECOBOT - VISAO", frame)

    # ESC = sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

print()
print("[ECOBOT] Sistema encerrado.")