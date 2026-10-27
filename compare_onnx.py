
import numpy as np
import onnxruntime as ort
import torch

from src.predict import load_model

model, mean, scale = load_model()
model.eval()

session = ort.InferenceSession(
    "models/emotion_classifier.onnx",
    providers=["CPUExecutionProvider"],
)

rng = np.random.default_rng(42)
features = rng.normal(size=(10, 80)).astype(np.float32)

# Run PyTorch on all 10 examples.
with torch.inference_mode():
    torch_logits = model(
        torch.from_numpy(features)
    ).numpy()

# ONNX currently accepts a batch size of 1.
# Run each example individually and combine the results.
onnx_logits = np.concatenate(
    [
        session.run(
            ["logits"],
            {"features": features[i:i + 1]}
        )[0]
        for i in range(len(features))
    ],
    axis=0,
)

max_difference = np.max(
    np.abs(torch_logits - onnx_logits)
)

torch_predictions = np.argmax(torch_logits, axis=1)
onnx_predictions = np.argmax(onnx_logits, axis=1)

print("Maximum logit difference:", max_difference)
print(
    "Matching predicted classes:",
    np.array_equal(torch_predictions, onnx_predictions),
)

np.testing.assert_allclose(
    torch_logits,
    onnx_logits,
    rtol=1e-4,
    atol=1e-5,
)

print("PASS: ONNX matches PyTorch.")

