
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from src.features import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_classifier.onnx"
SCALER_PATH = BASE_DIR / "models" / "emotion_scaler.npz"

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


@lru_cache(maxsize=1)
def load_onnx_model():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    with np.load(SCALER_PATH, allow_pickle=False) as scaler:
        mean = scaler["mean"]
        scale = scaler["scale"]

    return session, mean, scale


def predict_emotion(audio_path):
    start = perf_counter()

    session, mean, scale = load_onnx_model()

    features = extract_features(audio_path)
    features = (features - mean) / scale

    x = np.asarray(features, dtype=np.float32).reshape(1, 80)

    logits = session.run(
        ["logits"],
        {"features": x},
    )[0][0]

    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values)

    predicted_id = int(np.argmax(probabilities))

    print(
        f"[ONNX] Total prediction: "
        f"{perf_counter() - start:.3f}s",
        flush=True,
    )

    return EMOTIONS[predicted_id], probabilities
