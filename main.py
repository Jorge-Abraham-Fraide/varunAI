from flask import Flask, request, jsonify
import requests
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO

# Inicializar Flask
app = Flask(__name__)

# URL base de la API del microscopio OpenFlexure
api_url = 'http://10.10.40.18:5000/api/v2'

# Inicializar el modelo
model = YOLO('models/microplasticos1.pt')

def fetch_image_from_microscope(api_url):
    try:
        response = requests.get(api_url + '/streams/snapshot')
        response.raise_for_status()  # Lanza un error si la solicitud no fue exitosa
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

def process_image(frame):
    # Realizar la inferencia
    results = model.predict(frame, imgsz=640, conf=0.1)
    if len(results) != 0:
        microplastic_count = len(results[0].boxes)
    else:
        microplastic_count = 0
    
    return microplastic_count

@app.route('/capture_and_process', methods=['GET'])
def capture_and_process():
    frame = fetch_image_from_microscope(api_url)
    if frame is not None:
        microplastic_count = process_image(frame)
        
        response = {
            'microplastic_count': microplastic_count
        }
        
        return jsonify(response)
    else:
        return jsonify({'error': 'Failed to fetch image from microscope'}), 500

if __name__ == '__main__':
    # Ejecutar la aplicación Flask en HTTP
    app.run(host='0.0.0.0', port=5000)





