from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    print("Step 1: Training model...")
    train_model()

    print("\nStep 2: Evaluating model...")
    evaluate_model()