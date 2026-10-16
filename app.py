
import os
import tempfile
from time import perf_counter

import gradio as gr
import numpy as np
import soundfile as sf

from src.model import EMOTIONS
from src.predict import predict_emotion


def warm_up_model():
    """Warm up the complete prediction pipeline with synthetic audio."""
    start = perf_counter()

    sample_rate = 22050
    duration_seconds = 1.0
    samples = np.zeros(
        int(sample_rate * duration_seconds),
        dtype=np.float32
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as audio_file:
            temp_path = audio_file.name

        sf.write(temp_path, samples, sample_rate)

        predict_emotion(temp_path)

        print(
            f"[WARMUP] Completed in "
            f"{perf_counter() - start:.3f}s",
            flush=True
        )

    except Exception as error:
        print(
            f"[WARMUP] Failed: {error}",
            flush=True
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_emotion(audio_path):
    if audio_path is None:
        raise gr.Error(
            "Please upload an audio file or record your voice."
        )

    try:
        emotion, probabilities = predict_emotion(audio_path)
    except Exception as error:
        raise gr.Error(
            "Unable to analyze this recording. "
            "Try a short, clear speech recording."
        ) from error

    scores = {
        label: float(score)
        for label, score in zip(EMOTIONS, probabilities)
    }

    return emotion.capitalize(), scores


with gr.Blocks(title="Speech Emotion Recognition") as demo:
    gr.Markdown(
        "# 🎙️ Speech Emotion Recognition\n"
        "### Powered by PyTorch\n\n"
        "Upload a recording or use your microphone "
        "to explore how an AI model classifies vocal expressions."
    )

    with gr.Row():
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Your Speech Recording"
        )

    analyze_button = gr.Button(
        "Analyze Emotion",
        variant="primary"
    )

    predicted_emotion = gr.Textbox(
        label="Predicted Emotion",
        interactive=False
    )

    probabilities = gr.Label(
        label="Emotion Scores",
        num_top_classes=8
    )

    gr.Markdown(
        "**Important:** This is an experimental model "
        "trained on acted speech from the RAVDESS dataset. "
        "Its accuracy on held-out RAVDESS speakers was "
        "**39.58%**. Predictions do not reliably establish "
        "a person's actual feelings."
    )

    analyze_button.click(
        fn=analyze_emotion,
        inputs=audio,
        outputs=[predicted_emotion, probabilities]
    )


if __name__ == "__main__":
    if os.environ.get("ENABLE_WARMUP", "0") == "1":
        warm_up_model()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )

