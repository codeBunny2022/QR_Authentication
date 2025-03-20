import cv2
import joblib
import tensorflow as tf
import numpy as np
from feature_extraction import extract_hog_features, extract_lbp_features
from config import RF_MODEL_PATH, SVM_MODEL_PATH, CNN_MODEL_PATH

# Load models
rf_model = joblib.load(RF_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

def predict_qr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    # Extract features
    hog_feat = extract_hog_features(img)
    lbp_feat = extract_lbp_features(img)
    features = np.hstack([hog_feat, lbp_feat]).reshape(1, -1)

    rf_pred = rf_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]

    # Deep Learning Prediction
    cnn_input = img.reshape(1, 256, 256, 1) / 255.0
    cnn_pred = (cnn_model.predict(cnn_input) > 0.5).astype("int32")[0][0]

    return {"RandomForest": rf_pred, "SVM": svm_pred, "CNN": cnn_pred}

if __name__ == "__main__":
    test_image = "test_image.png"
    result = predict_qr(test_image)
    print(f"Prediction for {test_image}: {result}")
