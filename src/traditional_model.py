import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from feature_extraction import extract_features
from data_preprocessing import load_images, preprocess_dataset

# Load and preprocess data
images, labels = load_images()  # Remove the incorrect argument
images = preprocess_dataset(images)
features = extract_features(images)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train SVM Model
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Predictions
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_test)

# Evaluate Performance
print("Random Forest Performance:")
print(classification_report(y_test, rf_preds))

print("SVM Performance:")
print(classification_report(y_test, svm_preds))

# Save models
joblib.dump(rf_model, "../models/random_forest.pkl")
joblib.dump(svm_model, "../models/svm.pkl")
