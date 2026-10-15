import os
import gradio as gr

from src.model import EMOTIONS
from src.predict import predict_emotion


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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )
