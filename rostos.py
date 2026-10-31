import cv2

print("1 - OpenCV carregado")

arquivo = "haarcascade_frontalface_default.xml"

print("2 - Procurando classificador...")
print("Arquivo:", arquivo)

cascata = cv2.CascadeClassifier(arquivo)

if cascata.empty():
    print("ERRO: o classificador nao foi carregado.")
    exit()

print("3 - Classificador carregado!")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERRO: a webcam nao abriu.")
    exit()

print("4 - Webcam aberta!")
print("5 - Procurando rostos...")

while True:

    sucesso, frame = camera.read()

    if not sucesso:
        print("ERRO: nao consegui capturar a imagem.")
        break

    frame = cv2.flip(frame, 1)

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostos = cascata.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    print(f"\rRostos detectados: {len(rostos)}", end="")

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

    cv2.imshow("ECOBOT - Reconhecimento de Rostos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

print("\nECOBOT encerrado.")