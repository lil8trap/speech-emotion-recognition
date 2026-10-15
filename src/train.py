
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from src.model import EmotionClassifier

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "ravdess_features_80.npz"
MODEL_FILE = ROOT / "models" / "emotion_nn_reproduced.pt"

torch.manual_seed(42)
np.random.seed(42)


def main():
    data = np.load(DATA_FILE)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    model = EmotionClassifier()

    counts = np.bincount(y_train, minlength=8)
    weights = len(y_train) / (8 * counts)

    weights = torch.tensor(weights, dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    patience = 30
    wait = 0

    for epoch in range(300):
        model.train()

        logits = model(X_train)
        train_loss = loss_fn(logits, y_train_t)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val_t)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}: "
                f"Train={train_loss.item():.4f}, "
                f"Val={val_loss.item():.4f}"
            )

        if wait >= patience:
            print("Early stopping at epoch:", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predictions = model(X_val).argmax(dim=1).numpy()

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )

    print("\nBest epoch:", best_epoch)
    print("Validation accuracy:", round(accuracy, 4))
    print("Validation macro-F1:", round(macro_f1, 4))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model.network.state_dict(),
        "scaler_mean": torch.tensor(
            scaler.mean_, dtype=torch.float32
        ),
        "scaler_scale": torch.tensor(
            scaler.scale_, dtype=torch.float32
        ),
        "best_epoch": best_epoch,
        "class_weights": weights
    }, MODEL_FILE)

    print("\nReproduced checkpoint saved to:", MODEL_FILE)
    print("Original selected checkpoint remains unchanged.")


if __name__ == "__main__":
    main()
