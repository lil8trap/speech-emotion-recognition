
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.model import EmotionClassifier, EMOTIONS


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained emotion classifier"
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset partition to evaluate",
    )

    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "models" / "emotion_nn_reproduced.pt"
        ),
        help="Path to the model checkpoint",
    )

    args = parser.parse_args()

    data = np.load(
        ROOT / "ravdess_features_80.npz"
    )

    X = data[f"X_{args.split}"]
    y = data[f"y_{args.split}"]

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    mean = checkpoint["scaler_mean"].numpy()
    scale = checkpoint["scaler_scale"].numpy()

    X_scaled = (X - mean) / scale

    inputs = torch.tensor(
        X_scaled, dtype=torch.float32
    )

    model = EmotionClassifier()

    model.network.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    with torch.no_grad():
        logits = model(inputs)
        predictions = logits.argmax(dim=1).numpy()

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y, predictions, average="macro",
        zero_division=0,
    )

    print(f"\n=== {args.split.upper()} RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y,
        predictions,
        labels=range(8),
        target_names=EMOTIONS,
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(
        y, predictions, labels=range(8)
    ))


if __name__ == "__main__":
    main()
