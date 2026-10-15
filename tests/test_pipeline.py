
import unittest
from pathlib import Path

import numpy as np
import torch

from src.model import EmotionClassifier, EMOTIONS
from src.features import extract_features
from src.predict import load_model, predict_emotion

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_AUDIO = (
    ROOT
    / "data/ravdess/Audio_Speech_Actors_01-24"
    / "Actor_01"
    / "03-01-01-01-01-01-01.wav"
)

CHECKPOINT = (
    ROOT
    / "models"
    / "emotion_nn_80_weighted_checkpoint.pt"
)


class TestEmotionPipeline(unittest.TestCase):

    def test_eight_emotion_labels(self):
        self.assertEqual(len(EMOTIONS), 8)
        self.assertEqual(len(set(EMOTIONS)), 8)

    def test_model_output_shape(self):
        model = EmotionClassifier()
        sample = torch.randn(1, 80)
        output = model(sample)

        self.assertEqual(tuple(output.shape), (1, 8))

    def test_probabilities_sum_to_one(self):
        model = EmotionClassifier()
        sample = torch.randn(1, 80)

        with torch.no_grad():
            logits = model(sample)
            probabilities = torch.softmax(logits, dim=1)

        self.assertAlmostEqual(
            probabilities.sum().item(),
            1.0,
            places=5
        )

    def test_missing_audio_file(self):
        with self.assertRaises(FileNotFoundError):
            extract_features(
                ROOT / "nonexistent_audio_file.wav"
            )

    @unittest.skipUnless(
        SAMPLE_AUDIO.exists(),
        "RAVDESS sample audio not available"
    )
    def test_feature_extraction(self):
        features = extract_features(SAMPLE_AUDIO)

        self.assertEqual(features.shape, (80,))
        self.assertTrue(np.isfinite(features).all())

    @unittest.skipUnless(
        CHECKPOINT.exists(),
        "Trained checkpoint not available"
    )
    def test_checkpoint_loading(self):
        model, mean, scale = load_model()

        self.assertIsInstance(model, EmotionClassifier)
        self.assertEqual(mean.shape, (80,))
        self.assertEqual(scale.shape, (80,))

    @unittest.skipUnless(
        SAMPLE_AUDIO.exists() and CHECKPOINT.exists(),
        "Audio sample or checkpoint unavailable"
    )
    def test_end_to_end_prediction(self):
        emotion, probabilities = predict_emotion(
            SAMPLE_AUDIO
        )

        self.assertIn(emotion, EMOTIONS)
        self.assertEqual(probabilities.shape, (8,))
        self.assertAlmostEqual(
            float(probabilities.sum()),
            1.0,
            places=5
        )


if __name__ == "__main__":
    unittest.main()
