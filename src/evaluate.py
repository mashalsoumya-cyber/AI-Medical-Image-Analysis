import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from src.dataset import get_data_generators
from src.config import MODEL_PATH, CLASS_NAMES
from src.utils import save_confusion_matrix, plot_sample_predictions


def evaluate_model():
    _, _, test_generator = get_data_generators()

    model = tf.keras.models.load_model(MODEL_PATH)

    predictions = model.predict(test_generator, verbose=0)
    y_pred = (predictions > 0.5).astype("int32").flatten()
    y_true = test_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    save_confusion_matrix(y_true, y_pred, CLASS_NAMES)

    test_generator.reset()
    batch_images, batch_labels = next(test_generator)
    batch_preds = model.predict(batch_images, verbose=0)
    batch_preds = (batch_preds > 0.5).astype("int32").flatten()

    plot_sample_predictions(batch_images, batch_labels, batch_preds, CLASS_NAMES)


if __name__ == "__main__":
    evaluate_model()