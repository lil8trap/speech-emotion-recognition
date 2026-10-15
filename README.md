# Speech Emotion Recognition with PyTorch

An eight-class speech emotion recognition system built with PyTorch, Librosa, NumPy, and scikit-learn.

The project extracts acoustic features from speech recordings, trains neural-network classifiers, compares them against an SVM baseline, and evaluates performance on previously unseen speakers.

## Features

- Speech emotion classification across eight categories
- MFCC-based audio feature extraction
- PyTorch multilayer perceptron (MLP)
- Class-weighted cross-entropy loss
- Early stopping and model checkpointing
- Speaker-independent evaluation
- Command-line inference
- Automated pipeline tests

## Dataset

This project uses the RAVDESS speech dataset, consisting of 1,440 audio recordings from 24 actors.

The emotion categories are neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

To reduce speaker leakage, recordings were partitioned by actor:

| Partition | Actors | Recordings |
|---|---|---:|
| Training | 01–16 | 960 |
| Validation | 17–20 | 240 |
| Test | 21–24 | 240 |

The dataset is not distributed with this repository.

## Feature Engineering

Each recording is loaded at a sample rate of 22,050 Hz. Librosa extracts 40 Mel-frequency cepstral coefficients (MFCCs).

For each coefficient, its mean and standard deviation across time are calculated, producing 80 features per recording.

A StandardScaler is fitted using only the training partition. The resulting parameters are reused during validation, testing, and inference.

## Neural Network Architecture

The classifier is a fully connected multilayer perceptron:

```text
Input: 80 features
    ↓
Linear(80, 64)
    ↓
ReLU
    ↓
Linear(64, 32)
    ↓
ReLU
    ↓
Linear(32, 8)
    ↓
8 emotion logits
```

The selected model was trained with Adam, a learning rate of 0.001, class-weighted cross-entropy loss, and early stopping with patience of 30 epochs.

## Results

### Validation experiments

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| 40-feature linear SVM | 30.00% | 0.2611 |
| 40-feature PyTorch MLP | 30.00% | 0.2351 |
| 80-feature linear SVM | 40.42% | 0.3903 |
| 80-feature PyTorch MLP | 46.25% | 0.4075 |
| 80-feature weighted PyTorch MLP | 45.00% | 0.4246 |

The class-weighted model was selected based on validation macro-F1.

### Held-out test evaluation

The selected model was evaluated on 240 recordings from actors 21–24.

**Test accuracy: 39.58%**

**Test macro-F1: 0.3727**

Performance varied substantially across emotion classes. The model achieved its strongest recall for surprised speech and struggled most with sad speech.

These results are from one speaker-independent partition, not repeated cross-validation.

## Limitations

- The RAVDESS dataset contains acted emotions rather than spontaneous everyday speech.
- Averaged MFCC statistics discard temporal ordering.
- Performance varies across speakers and emotion classes.
- Softmax probabilities are not necessarily calibrated confidence estimates.
- Performance on the RAVDESS dataset does not establish reliability on real-world recordings.

## Installation

The project was developed using Python 3.14 on macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded dependency versions are from the development environment; fresh-environment installation remains to be verified.

## Running Predictions

After placing the trained checkpoint at `models/emotion_nn_80_weighted_checkpoint.pt`, run:

```bash
python -m src.predict path/to/audio.wav
```

The program displays the predicted emotion and softmax scores for all eight categories.

The trained checkpoint is not currently distributed with the repository.

## Automated Tests

Run:

```bash
python -m unittest discover -s tests -v
```

Five automated tests cover model output dimensions, emotion labels, feature extraction, checkpoint loading, and end-to-end inference.

The inference-related tests require access to the checkpoint and a sample RAVDESS recording. The dataset and checkpoint are excluded from version control.

## Future Work

- Consolidate training and evaluation into reusable scripts.
- Compare performance across multiple speaker-independent splits.
- Experiment with temporal audio architectures such as convolutional networks.
- Improve reproducibility and checkpoint distribution.
- Investigate model calibration and robustness on external speech data.

## Technology Stack

Python, PyTorch, Librosa, NumPy, scikit-learn, SoundFile.

## Project Status

Research prototype with completed held-out evaluation, working local inference, and basic automated tests. Reproducibility and repository packaging are ongoing.
