
import os, io, time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

MODEL_PATH  = "/content/models/best_model_final.keras"
IMG_SIZE    = (224, 224)
THRESHOLD   = 0.5
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "classes": CLASS_NAMES})

@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": model.name,
        "parameters": model.count_params(),
        "classes": CLASS_NAMES
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file  = request.files["file"]
    start = time.time()
    arr   = preprocess(file.read())
    prob  = float(model.predict(arr, verbose=0)[0][0])
    pred  = CLASS_NAMES[1] if prob >= THRESHOLD else CLASS_NAMES[0]
    conf  = prob if pred == "PNEUMONIA" else (1 - prob)
    return jsonify({
        "prediction": pred,
        "confidence": round(conf, 4),
        "probabilities": {"NORMAL": round(1-prob,4), "PNEUMONIA": round(prob,4)},
        "time_ms": round((time.time()-start)*1000, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
