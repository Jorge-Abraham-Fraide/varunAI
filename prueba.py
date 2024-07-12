import requests
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO

def url_to_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# URL de la imagen en línea
image_url = 'https://civiclaboratory.nl/wp-content/uploads/2021/05/egccqo8woaegb3x.jpg?w=768'

# Cargar la imagen desde la URL
frame = url_to_image(image_url)

# Inicializar el modelo
model = YOLO('models/microplasticos1.pt')

# Realizar la inferencia
results = model.predict(frame, imgsz=640, conf=0.1)
if len(results) != 0:
    for res in results:
        print('Microplastics detect')

    annotated_frames = results[0].plot()

# Mostrar el resultado
cv2.imshow('Microplastic detect', annotated_frames)
cv2.waitKey(0)  # Esperar a que se pulse una tecla
cv2.destroyAllWindows()
