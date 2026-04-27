# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Initialize Flask, configuring it to serve static files from the current directory
app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Resolve absolute paths for reliable model loading in production
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading models...")
binary_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "best_model_colab.h5"))
multiclass_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "multiclass_model_v2.h5"))
segmentation_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "unet_segmentation.h5"))
print("✅ Models loaded")

def preprocess_224(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_128(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=(0, -1))

# Serve the frontend UI
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()

        # Phase 1: Binary classification
        input_224 = preprocess_224(image_bytes)
        binary_pred = binary_model.predict(input_224)[0][0]
        is_tumor = bool(binary_pred > 0.5)
        detection_conf = float(binary_pred) if is_tumor else float(1 - binary_pred)

        if not is_tumor:
            return jsonify({'isTumor': False, 'detectionConf': detection_conf})

        # Phase 3A: Multi-class
        multi_pred = multiclass_model.predict(input_224)[0]
        type_probs = multi_pred.tolist()
        tumor_type = ['Glioma', 'Meningioma', 'Pituitary'][np.argmax(multi_pred)]

        # Phase 3B: Segmentation
        input_128 = preprocess_128(image_bytes)
        seg_mask = segmentation_model.predict(input_128)[0, :, :, 0]
        tumor_pixels = np.sum(seg_mask > 0.5)
        gray = input_128[0, :, :, 0]
        brain_pixels = np.sum(gray > 0.1)
        size_percent = (tumor_pixels / brain_pixels) * 100 if brain_pixels > 0 else 0

        # Simulate hemisphere
        hemisphere = "LEFT" if np.random.rand() > 0.5 else "RIGHT"

        return jsonify({
            'isTumor': True,
            'detectionConf': detection_conf,
            'typeProbs': type_probs,
            'tumorType': str(tumor_type),
            'hemisphere': hemisphere,
            'sizePercent': float(size_percent),
            'pixelCount': int(tumor_pixels)
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    # Cloud environments inject the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
