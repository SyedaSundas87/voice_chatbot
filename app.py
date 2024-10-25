import os
from dotenv import load_dotenv
import whisper
from gtts import gTTS
from tempfile import NamedTemporaryFile
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from the environment
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not set. Please make sure it's defined in the .env file.")
    st.stop()

# Initialize Whisper model for transcription
whisper_model = whisper.load_model("base")

# Initialize Groq client (you will need to ensure this library supports Streamlit)
from groq import Groq
client = Groq(api_key=api_key)

# Function to handle voice input, generate response, and convert to speech
def voice_chatbot(audio_file):
    # Step 1: Transcribe the audio input using Whisper
    transcription = whisper_model.transcribe(audio_file)["text"]

    # Step 2: Use the transcription to generate a response from the LLM
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": transcription}],
        model="llama-3.1-70b-versatile"  # Example model, change to the one you're using
    )
    response_text = chat_completion.choices[0].message.content

    # Step 3: Convert the text response to speech using gTTS
    tts = gTTS(response_text)
    temp_audio_file = NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_audio_file.name)

    return response_text, temp_audio_file.name

# Streamlit UI
st.title("Voice-to-Voice Chatbot 💬🎤")

uploaded_audio = st.file_uploader("Upload your voice input", type=["wav", "mp3", "ogg"])

if uploaded_audio is not None:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_audio.read())
        temp_audio_path = temp_file.name
    
    # Get the chatbot response
    response_text, response_audio = voice_chatbot(temp_audio_path)

    # Display the text response
    st.subheader("Generated Response")
    st.write(response_text)

    # Play the audio response
    st.subheader("Response as Audio")
    audio_file = open(response_audio, 'rb')
    st.audio(audio_file.read(), format="audio/mp3")

# Instructions for setup on GitHub
st.markdown("""
### To deploy this app on Streamlit through GitHub:
1. **Create a GitHub Repository**: Push this Python code and the `.env` file (excluding the sensitive API keys) to a new GitHub repo.
2. **Create a `requirements.txt` file**: Include the necessary packages like this:

