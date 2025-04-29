import os
from groq import Groq

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_file(file_path):
    """
    Transcribe the audio file using Groq's Whisper model.
    
    :param file_path: Path to the audio file.
    :return: Transcription text.
    """
    with open(file_path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            file=(os.path.basename(file_path), f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    return resp.text.strip()

# Specify the file path
file_path = "Call recording Sujan.P Hackaton Ait_250429_084023.m4a"

# Transcribe the file
transcription = transcribe_file(file_path)
print("Transcription:")
print(transcription)