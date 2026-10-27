
from pathlib import Path

import numpy as np
import torch

from src.predict import load_model


OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

model, scaler_mean, scaler_scale = load_model()

model.eval()

dummy_input = torch.randn(1, 80, dtype=torch.float32)

onnx_path = OUTPUT_DIR / "emotion_classifier.onnx"

torch.onnx.export(
    model,
    (dummy_input,),
    str(onnx_path),
    input_names=["features"],
    output_names=["logits"],
    opset_version=18,
    dynamo=True,
)

np.savez(
    OUTPUT_DIR / "emotion_scaler.npz",
    mean=scaler_mean,
    scale=scaler_scale,
)

print("ONNX model exported:", onnx_path)
print("Model size:", round(onnx_path.stat().st_size / 1024, 1), "KB")

