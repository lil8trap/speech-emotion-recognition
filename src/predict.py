from time import perf_counter
import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.model import EmotionClassifier, EMOTIONS
from src.features import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "emotion_nn_80_weighted_checkpoint.pt"
)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained classifier once per Python process."""

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = EmotionClassifier()

    model.network.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    scaler_mean = checkpoint["scaler_mean"].numpy()
    scaler_scale = checkpoint["scaler_scale"].numpy()

    return model, scaler_mean, scaler_scale


def predict_emotion(audio_path):
    """Predict an emotion and measure processing time."""

    total_start = perf_counter()

    # Step 1: Load or retrieve cached model
    start = perf_counter()
    model, mean, scale = load_model()
    model_time = perf_counter() - start

    # Step 2: Extract acoustic features
    start = perf_counter()
    features = extract_features(audio_path)
    feature_time = perf_counter() - start

    # Step 3: Scale features and run inference
    start = perf_counter()

    features = (features - mean) / scale

    x = torch.as_tensor(
        np.asarray(features, dtype=np.float32)
    ).unsqueeze(0)

    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = int(probabilities.argmax().item())
    result = probabilities.numpy()

    inference_time = perf_counter() - start
    total_time = perf_counter() - total_start

    print(
        f"[PERFORMANCE] Model loading: {model_time:.3f}s",
        flush=True
    )
    print(
        f"[PERFORMANCE] Feature extraction: {feature_time:.3f}s",
        flush=True
    )
    print(
        f"[PERFORMANCE] Inference: {inference_time:.3f}s",
        flush=True
    )
    print(
        f"[PERFORMANCE] Total prediction: {total_time:.3f}s",
        flush=True
    )

    return EMOTIONS[predicted_id], result



def main():
    parser = argparse.ArgumentParser(
        description="Predict speech emotions using PyTorch"
    )

    parser.add_argument(
        "audio",
        help="Path to a WAV audio recording"
    )

    args = parser.parse_args()

    start = perf_counter()

    emotion, probabilities = predict_emotion(args.audio)

    elapsed = perf_counter() - start

    print("\nPredicted emotion:", emotion)

    print("\nEmotion probabilities:")

    for label, probability in zip(
        EMOTIONS, probabilities
    ):
        print(f"{label}: {probability * 100:.2f}%")

    print(f"\nPrediction time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

