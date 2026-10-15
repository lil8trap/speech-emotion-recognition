
import torch
from torch import nn


class EmotionClassifier(nn.Module):
    """
    Neural network for eight-class
    speech emotion classification.

    Input: 80 MFCC features
    Output: 8 emotion logits
    """

    def __init__(self, input_size=80, num_classes=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]
