
from pathlib import Path

import librosa
import numpy as np


def extract_features(audio_path):
    """
    Extract 80 MFCC-derived features from audio.

    Returns:
        NumPy array with shape (80,)
    """

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio, sr = librosa.load(
        audio_path,
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features.astype(np.float32)
