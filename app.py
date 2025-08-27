# back/app.py  (ONLY add the /health route below; keep your existing code)

from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import io
from tensorflow.keras.models import load_model
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Load the updated segmentation model
segmentation_model = load_model('unet_full_model.keras', compile=False)

# Load the Hugging Face classification model
processor = AutoImageProcessor.from_pretrained("ALM-AHME/convnextv2-large-1k-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20-V2")
model = AutoModelForImageClassification.from_pretrained("ALM-AHME/convnextv2-large-1k-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20-V2")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 Megabytes

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

# Function to preprocess image for model input
def preprocess_image(image, target_size=(256, 384)):
    """ Applies hair removal preprocessing before feeding to the model. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Blackhat morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Create mask using thresholding
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Apply inpainting to remove hair
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    # Convert to grayscale and resize
    inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
    inpainted_image = cv2.resize(inpainted_image, (target_size[1], target_size[0])) / 255.0

    # Expand dimensions for model input
    preprocessed_image = np.expand_dims(inpainted_image, axis=0)  # batch
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)  # channel
    return image, inpainted_image, preprocessed_image

@app.post('/predict')
def predict_segmentation():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        image = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            return "Image could not be decoded", 400

        _, processed_image, preprocessed_image = preprocess_image(image)

        prediction = segmentation_model.predict(preprocessed_image)
        predicted_mask = (prediction.squeeze() * 255).astype(np.uint8)

        _, encoded_img = cv2.imencode('.png', predicted_mask)
        response_data = io.BytesIO(encoded_img)
        response_data.seek(0)

        return send_file(response_data, mimetype='image/png')
    except Exception as e:
        print(f"Error during segmentation prediction: {e}")
        return "An internal server error occurred", 500

@app.post('/classify')
def predict_classification():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        image = Image.open(file.stream).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        id2label = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
        class_name = id2label.get(predicted_class_idx, "Unknown")
        return jsonify({"class": class_name})
    except Exception as e:
        print(f"Error during classification prediction: {e}")
        return "An internal server error occurred", 500

if __name__ == '__main__':
    # For local debug-only; container uses gunicorn CMD
    app.run(debug=True, host='0.0.0.0', port=5000)
