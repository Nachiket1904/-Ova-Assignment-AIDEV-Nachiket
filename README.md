# Systemic Altruism AI Assistant

This project is a part of the Systemic Altruism AI Development Internship. It showcases the development of a multimodal AI assistant capable of processing both speech and image inputs to generate meaningful text and audio outputs. Leveraging cutting-edge AI technologies, this assistant demonstrates the integration of speech recognition, image description, and text-to-speech capabilities.

## Overview

The AI assistant is designed to accept an audio input and an image file through a user-friendly web interface. It transcribes the audio input to text, utilizes this text to generate a detailed description of the provided image, and finally, converts this description back into audio output. This process showcases the assistant's ability to understand and generate multimodal content, making it a versatile tool for various applications.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**

```
git clone https://your-repo-link.git
cd your-repo-directory
```
Install dependencies
Ensure you have Python 3.6+ installed on your system. Then, install the required packages using the following command:

```pip install -r requirements.txt```

Run the application
```python app.py```

Usage
Once the application is running, you will be presented with a web interface where you can:

Record or upload an audio file.
Upload an image file.
Submit the inputs for processing by the AI assistant.
The assistant will then display the transcribed text, generate a detailed description of the image based on the audio input, and provide an audio output of the image description.

Technologies
This project utilizes several key technologies:

PyTorch & Transformers: For loading and running the AI models.
Whisper: For speech-to-text transcription.
Pillow (PIL): For image processing.
GTTS (Google Text-to-Speech): For converting text descriptions into audio.
Gradio: To create the interactive web interface.
