import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io

st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

with tab2:

    # ------------------ TEXT TO SPEECH ------------------
    st.header("🗣️ Text to Speech")
    text = st.text_area("Enter text to convert to speech:")

    if st.button("Convert to Audio"):
        if text.strip():
            tts = gTTS(text, lang='en')
            tts.save("output.mp3")
            audio_file = open("output.mp3", "rb")
            st.audio(audio_file.read(), format='audio/mp3')
            st.success("✅ Conversion complete!")
        else:
            st.warning("Please enter some text.")

    
    # ------------------ SPEECH TO TEXT ------------------
    st.header("🗣️ Speech to Text")

    # Upload audio
    uploaded_audio = st.file_uploader("Upload audio file (wav, mp3, m4a)", type=["wav","mp3","m4a"])

    if uploaded_audio:
        # Convert uploaded audio to PCM WAV
        audio_bytes = uploaded_audio.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Play audio in Streamlit
        st.audio(wav_io, format="audio/wav")

        if st.button("Transcribe Audio"):
            recognizer = sr.Recognizer()
            # SpeechRecognition requires a real file-like object, so we reset BytesIO
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)

            with st.spinner("Transcribing..."):
                try:
                    text_output = recognizer.recognize_google(audio_data)
                    st.success("✅ Transcription complete!")
                    st.subheader("Transcribed Text")
                    st.write(text_output)
                except sr.UnknownValueError:
                    st.error("Speech not recognized.")
                except sr.RequestError:
                    st.error("Google API unavailable or network error.")