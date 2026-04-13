from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

from src.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = prepare_image(filepath)
            pred = model.predict(image)[0][0]

            if pred > 0.5:
                prediction = CLASS_NAMES[1]
                confidence = round(float(pred) * 100, 2)
            else:
                prediction = CLASS_NAMES[0]
                confidence = round(float((1 - pred)) * 100, 2)

            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)