
import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter

import gradio as gr
import librosa
import numpy as np
import soundfile as sf

from src.model import EMOTIONS
from src.predict import predict_emotion


# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_DURATION_SECONDS = 30
MAX_FILE_BYTES = 25 * 1024 * 1024


# --------------------------------------------------
# Optional model warm-up
# --------------------------------------------------

def warm_up_model():
    """Optionally initialize the prediction pipeline at startup."""

    start = perf_counter()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as audio_file:
            temp_path = audio_file.name

        sf.write(
            temp_path,
            np.zeros(22050, dtype=np.float32),
            22050
        )

        predict_emotion(temp_path)

        logger.info(
            "[WARMUP] Completed in %.3fs",
            perf_counter() - start
        )

    except Exception:
        logger.exception("[WARMUP] Failed")

    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


# --------------------------------------------------
# Prediction handler
# --------------------------------------------------

def analyze_emotion(audio_path, progress=gr.Progress()):
    """Validate an audio recording and predict its emotion."""

    if not audio_path:
        raise gr.Error(
            "Please record your voice or upload an audio file."
        )

    path = Path(audio_path)

    if not path.is_file():
        raise gr.Error(
            "The selected audio file could not be found."
        )

    if path.stat().st_size > MAX_FILE_BYTES:
        raise gr.Error(
            "This recording is too large. "
            "Please use a file smaller than 25 MB."
        )

    try:
        progress(0.1, desc="Checking recording")

        duration = librosa.get_duration(path=str(path))

        if duration <= 0:
            raise gr.Error(
                "This recording appears to be empty."
            )

        if duration > MAX_DURATION_SECONDS:
            raise gr.Error(
                "Please use a recording of 30 seconds or less."
            )

        progress(0.3, desc="Extracting audio features")

        emotion, probabilities = predict_emotion(path)

        scores = {
            label: float(score)
            for label, score in zip(EMOTIONS, probabilities)
        }

        progress(1.0, desc="Analysis complete")

        return emotion.capitalize(), scores

    except gr.Error:
        raise

    except Exception:
        logger.exception("Audio analysis failed")

        raise gr.Error(
            "Unable to analyze this recording. "
            "Please try a short, clear speech recording."
        )


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    title="Speech Emotion Recognition"
) as demo:

    gr.Markdown(
        """
        # 🎙️ Speech Emotion Recognition

        ### Audio Classification with PyTorch

        Explore how a neural network classifies emotional
        expressions in speech.

        **Instructions:**
        Record yourself speaking or upload an audio file,
        then click **Analyze Emotion**.

        Recommended recording length: **3–10 seconds**.

        Maximum recording length: **30 seconds**.
        """
    )

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
        """
        ---

        ### Understanding the Results

        The predicted emotion is the class with the
        highest model score.

        The chart displays the model's relative scores
        across eight emotion categories.

        A higher score does not guarantee a correct prediction.

        ### Important Model Limitations

        This experimental model was trained on acted speech
        from the RAVDESS dataset.

        Its accuracy on held-out RAVDESS speakers was
        **39.58%**.

        Predictions do not reliably determine a person's
        actual feelings.

        **Hosting note:** The free server may take time
        to wake up after inactivity.
        """
    )

    analyze_button.click(
        fn=analyze_emotion,
        inputs=audio,
        outputs=[
            predicted_emotion,
            probabilities
        ],
        show_progress="full"
    )


# --------------------------------------------------
# Application startup
# --------------------------------------------------

if __name__ == "__main__":

    if os.environ.get("ENABLE_WARMUP", "0") == "1":
        warm_up_model()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", "7860")
        )
    )

