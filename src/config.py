import os

# Define dataset paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FIRST_PRINT_DIR = os.path.join(DATA_DIR, "First Print")
SECOND_PRINT_DIR = os.path.join(DATA_DIR, "Second Print")

# Image Processing
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# Model Paths
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")
