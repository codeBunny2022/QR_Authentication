import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure 'src/' is in Python's module path BEFORE importing config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Now import modules
from feature_extraction import extract_hog_features, extract_lbp_features
from config import FIRST_PRINT_DIR, SECOND_PRINT_DIR, RF_MODEL_PATH, SVM_MODEL_PATH, CNN_MODEL_PATH

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
rf_model = joblib.load(RF_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

def predict_qr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    # Extract features for traditional ML models
    hog_feat = extract_hog_features(img)
    lbp_feat = extract_lbp_features(img)
    features = np.hstack([hog_feat, lbp_feat]).reshape(1, -1)

    rf_pred = rf_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]

    # CNN Prediction
    cnn_input = img.reshape(1, 256, 256, 1) / 255.0
    cnn_pred = (cnn_model.predict(cnn_input) > 0.5).astype("int32")[0][0]

    return {"RandomForest": int(rf_pred), "SVM": int(svm_pred), "CNN": int(cnn_pred)}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    result = predict_qr(file_path)
    os.remove(file_path)  # Cleanup

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
