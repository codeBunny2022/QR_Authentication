import numpy as np
import joblib
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from feature_extraction import extract_features
from data_preprocessing import load_images, preprocess_dataset
from config import RF_MODEL_PATH, SVM_MODEL_PATH, CNN_MODEL_PATH

# Load dataset
images, labels = load_images()
images = preprocess_dataset(images)
features = extract_features(images)

# Load trained models
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Predictions
rf_preds = rf_model.predict(features)
svm_preds = svm_model.predict(features)
cnn_preds = (cnn_model.predict(images.reshape(-1, 256, 256, 1) / 255.0) > 0.5).astype("int32").flatten()

# Print reports
print("Random Forest Report:\n", classification_report(labels, rf_preds))
print("SVM Report:\n", classification_report(labels, svm_preds))
print("CNN Report:\n", classification_report(labels, cnn_preds))

# Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["First Print", "Second Print"],
                yticklabels=["First Print", "Second Print"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

plot_confusion_matrix(labels, rf_preds, "Random Forest Confusion Matrix")
plot_confusion_matrix(labels, svm_preds, "SVM Confusion Matrix")
plot_confusion_matrix(labels, cnn_preds, "CNN Confusion Matrix")
