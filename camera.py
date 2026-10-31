import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: não foi possível abrir a webcam.")
    exit()

print("Webcam funcionando! Pressione ESC para sair.")

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem.")
        break

    cv2.imshow("ECOBOT - Visao Computacional", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()