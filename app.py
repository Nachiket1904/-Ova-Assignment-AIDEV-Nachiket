# app.py
import warnings
import os
import datetime
import re
from PIL import Image
import numpy as np
import torch
from transformers import pipeline, BitsAndBytesConfig
import whisper
import gradio as gr
from gtts import gTTS

# Suppress warnings
warnings.filterwarnings("ignore")

# Check if GPU is available, and set device accordingly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

# Load Whisper model for audio processing
whisper_model = whisper.load_model("medium", device=DEVICE)

# Initialize variables for text-to-image model if GPU is available
if torch.cuda.is_available():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs = {"quantization_config": quant_config}
else:
    model_kwargs = {}

# Load the text-to-image model with quantization if applicable
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline(
    "image-to-text",
    model=model_id,
    model_kwargs=model_kwargs
)

def transcribe(audio_path):
    # Function to transcribe audio file to text
    if audio_path is None or audio_path == '':
        return ''
    
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    
    return result.text

def img2txt(speech_text, input_image_path):
    # Function to generate text description from an image
    if not os.path.exists(input_image_path):
        return "No image provided."
    
    image = Image.open(input_image_path)
    prompt_instructions = f"""
    Given the input speech: {speech_text}. Describe the image using as much detail as possible,
    is it a painting, a photograph, what colors are predominant, what is the image about?
    Now generate a helpful answer.
    """
    prompt = f"USER: <image>\n{prompt_instructions}\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
    
    if outputs and len(outputs[0]["generated_text"]) > 0:
        return outputs[0]["generated_text"]
    else:
        return "No response generated."

def text_to_speech(text, file_path="output.mp3"):
    # Function to convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(file_path)
    return file_path

def process_inputs(audio_path, image_path):
    # Main function to process both audio and image inputs
    speech_to_text_output = transcribe(audio_path)
    chatgpt_output = img2txt(speech_to_text_output, image_path) if image_path else "No image provided."
    processed_audio_path = text_to_speech(chatgpt_output)
    
    return speech_to_text_output, chatgpt_output, processed_audio_path

# Setup Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="AI Output"),
        gr.Audio(label="Response Audio")
    ],
    title="LLM AI powered voice assistant for multimodal data",
    description="Upload an image and interact via voice input for an audio response."
)

if __name__ == "__main__":
    iface.launch(debug=True)
