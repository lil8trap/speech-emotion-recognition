
from pathlib import Path
from collections import Counter

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data/ravdess/Audio_Speech_Actors_01-24"
OUTPUT_FILE = ROOT / "ravdess_features_80.npz"

EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]


def main():
    files = sorted(DATA_DIR.rglob("*.wav"))

    if not files:
        raise FileNotFoundError(
            f"No WAV files found in {DATA_DIR}"
        )

    print(f"Found {len(files)} audio recordings")

    features = []
    labels = []
    actors = []

    for i, path in enumerate(files):
        parts = path.stem.split("-")

        emotion_id = int(parts[2]) - 1
        actor_id = int(parts[6])

        audio, sr = librosa.load(path, sr=22050)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        feature_vector = np.concatenate([
            mfcc_mean,
            mfcc_std
        ])

        features.append(feature_vector)
        labels.append(emotion_id)
        actors.append(actor_id)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} recordings")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    actors = np.asarray(actors, dtype=np.int64)

    train_mask = actors <= 16
    val_mask = (actors >= 17) & (actors <= 20)
    test_mask = (actors >= 21) & (actors <= 24)

    assert not (
        set(actors[train_mask]) & set(actors[val_mask])
    )
    assert not (
        set(actors[train_mask]) & set(actors[test_mask])
    )
    assert not (
        set(actors[val_mask]) & set(actors[test_mask])
    )

    print("\nDataset shapes:")
    print("Training:", X[train_mask].shape)
    print("Validation:", X[val_mask].shape)
    print("Testing:", X[test_mask].shape)

    print("\nEmotion distribution:")
    for label, count in sorted(Counter(y).items()):
        print(f"{EMOTIONS[label]}: {count}")

    np.savez(
        OUTPUT_FILE,
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_val=X[val_mask],
        y_val=y[val_mask],
        X_test=X[test_mask],
        y_test=y[test_mask]
    )

    print(f"\nSaved dataset to {OUTPUT_FILE}")
    print("Feature extraction completed!")


if __name__ == "__main__":
    main()
