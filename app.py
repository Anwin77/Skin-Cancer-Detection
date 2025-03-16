from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from PIL import Image
import io
import base64
import cv2
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock
from vit_keras import vit

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

custom_objects = {
    'ClassToken': ClassToken,
    'AddPositionEmbs': AddPositionEmbs,
    'TransformerBlock': TransformerBlock
}
model_path = 'vit_model.keras'
if not os.path.exists(model_path):
	raise FileNotFoundError(f"Model file {model_path} not found in {os.getcwd()}")
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)

def preprocess_image(image_data):
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (224, 224))
    img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        image_data = data['image']
        input_data = preprocess_image(image_data)
        prediction = model.predict(input_data)[0]
        class_labels = ['akiec: Actinic keratoses', 'bcc: Basal cell carcinoma', 'bkl: Benign keratosis-like lesions', 'df: Dermatofibroma', 'mel: Melanoma (most dangerous)', 'nv: Melanocytic nevi', 'vasc: Vascular lesions']  
        result = {
            'probabilities': {label: float(prob) for label, prob in zip(class_labels, prediction)},
            'predicted_class': class_labels[np.argmax(prediction)]
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Skin Cancer Detection API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
