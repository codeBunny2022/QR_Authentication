import cv2
import numpy as np
import os
from glob import glob
from config import FIRST_PRINT_DIR, SECOND_PRINT_DIR

def load_images():
    images = []
    labels = []
    print("Loading images...")

    # Check if directories exist
    if not os.path.exists(FIRST_PRINT_DIR) or not os.path.exists(SECOND_PRINT_DIR):
        print("Error: Dataset directories not found!")
        return np.array([]), np.array([])

    for label, category in enumerate([FIRST_PRINT_DIR, SECOND_PRINT_DIR]):
        image_paths = glob(os.path.join(category, "*.*"))  # Load all image types
        print(f"Found {len(image_paths)} images in {category}.")

        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Skipping unreadable file {image_path}")
                continue
            img = cv2.resize(img, (256, 256))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def preprocess_image(img):
    """ Apply preprocessing steps to an image. """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)
    return img

def preprocess_dataset(images):
    """ Preprocess entire dataset. """
    return np.array([preprocess_image(img) for img in images])

if __name__ == "__main__":
    images, labels = load_images()
    images = preprocess_dataset(images)
    print(f"Processed {len(images)} images successfully.")
