
from pathlib import Path
from time import perf_counter

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

    start = perf_counter()

    audio, sr = librosa.load(
        audio_path,
        sr=22050
    )

    load_time = perf_counter() - start

    start = perf_counter()

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_time = perf_counter() - start

    start = perf_counter()

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    features = features.astype(np.float32)

    statistics_time = perf_counter() - start

    print(
        f"[AUDIO] Loading/resampling: {load_time:.3f}s",
        flush=True
    )
    print(
        f"[AUDIO] MFCC computation: {mfcc_time:.3f}s",
        flush=True
    )
    print(
        f"[AUDIO] Statistics: {statistics_time:.3f}s",
        flush=True
    )

    return features

