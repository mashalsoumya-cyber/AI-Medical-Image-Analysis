from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import get_data_generators
from src.model import build_model
from src.config import MODEL_PATH, EPOCHS
from src.utils import plot_training_history


def train_model():
    print("Loading dataset...")

    train_generator, val_generator, _ = get_data_generators()

    print("Dataset loaded successfully")

    model = build_model()

    print("Model built successfully")

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting training...")

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_training_history(history)

    print(f"\nTraining completed. Model saved at: {MODEL_PATH}")