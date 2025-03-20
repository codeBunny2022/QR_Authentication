import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from data_preprocessing import load_images, preprocess_dataset
from joblib import Parallel, delayed

def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_lbp_features(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    return hist

def extract_features(images):
    hog_features = Parallel(n_jobs=-1)(delayed(extract_hog_features)(img) for img in images)
    lbp_features = Parallel(n_jobs=-1)(delayed(extract_lbp_features)(img) for img in images)
    return np.hstack([hog_features, lbp_features])

if __name__ == "__main__":
    images, labels = load_images()
    images = preprocess_dataset(images)
    features = extract_features(images)
    print(f"Extracted {features.shape[1]} features from {len(images)} images.")
