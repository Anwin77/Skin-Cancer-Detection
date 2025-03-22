import os
import requests
import tensorflow as tf
import tensorflow_addons as tfa
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import vit_keras.vit
import io

app = Flask(__name__)
CORS(app)

# Path to save the model
MODEL_PATH = "vit_model.h5"


# Load the model
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
		"Addons>FocalLoss": tfa.losses.SigmoidFocalCrossEntropy(),
		"ExtractToken": lambda x: x[:,0]
	},
	safe_mode=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise e

# Routes
@app.route('/')
def home():
    return "Skin Cancer Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])

        # Map the predicted class to a label (adjust based on your classes)
        class_labels = ['akiec: Actinic keratoses', 'bcc: Basal cell carcinoma', 'bkl: Benign keratosis-like lesions', 'df: Dermatofibroma', 'mel: Melanoma (most dangerous)', 'nv: Melanocytic nevi', 'vase: Vascular lesions']  # Replace with your actual labels
        result = {
            'predicted_class': class_labels[predicted_class],
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
