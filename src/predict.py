import cv2
import numpy as np
import tensorflow as tf
from src.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image. Check the file path.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(image_path):
    image = preprocess_image(image_path)
    pred = model.predict(image, verbose=0)[0][0]
    label = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
    confidence = float(pred if pred > 0.5 else 1 - pred)
    return label, confidence


if __name__ == "__main__":
    sample_path = "data/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    label, confidence = predict_image(sample_path)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")