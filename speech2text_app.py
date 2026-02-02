import torch
import gradio as gr
from transformers import pipeline


# -------- Speech to Text Function --------
def transcript_audio(audio_file):
    # Initialize Whisper speech-to-text pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30
    )

    # Transcribe audio
    result = pipe(audio_file, batch_size=8)["text"]
    return result


# -------- Gradio Interface --------
audio_input = gr.Audio(
    sources="upload",
    type="filepath",
    label="Upload Audio File"
)

output_text = gr.Textbox(
    label="Transcribed Text"
)

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Speech to Text App",
    description="Upload an audio file to convert speech into text"
)

# -------- Launch App --------
iface.launch(
    server_name="0.0.0.0",
    server_port=7860
)
