import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from config import DATA_DIR, CNN_MODEL_PATH

# Improved Model with Batch Normalization
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Improved Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)

# Train CNN
model = create_cnn_model()
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[lr_scheduler])

# Save model
model.save(CNN_MODEL_PATH)
print(f"Model saved to {CNN_MODEL_PATH}")
