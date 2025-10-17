import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
from PIL import Image, ImageStat
import numpy as np
from collections import Counter
import re
from textblob import TextBlob
import nltk

# Download required NLTK data (only once, but safe to run multiple times)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ------------------ IMAGE ANALYSIS TAB ------------------
with tab1:
    st.header("🖼️ Image Analysis")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"])
    
    if uploaded_image:
        # Load and display image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Basic characteristics
                st.subheader("📊 Basic Characteristics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Format", image.format if image.format else "Unknown")
                    st.metric("Width", f"{image.width} px")
                
                with col2:
                    st.metric("Mode", image.mode)
                    st.metric("Height", f"{image.height} px")
                
                with col3:
                    # File size
                    uploaded_image.seek(0)
                    file_size = len(uploaded_image.read()) / 1024  # KB
                    st.metric("File Size", f"{file_size:.2f} KB")
                    st.metric("Aspect Ratio", f"{image.width/image.height:.2f}")
                
                # Color Analysis
                st.subheader("🎨 Color Analysis")
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    rgb_image = image.convert('RGB')
                else:
                    rgb_image = image
                
                # Get image statistics
                stat = ImageStat.Stat(rgb_image)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Average Colors (RGB):**")
                    st.write(f"- Red: {stat.mean[0]:.2f}")
                    st.write(f"- Green: {stat.mean[1]:.2f}")
                    st.write(f"- Blue: {stat.mean[2]:.2f}")
                    
                    # Calculate brightness
                    brightness = sum(stat.mean[:3]) / 3
                    st.metric("Average Brightness", f"{brightness:.2f}/255")
                
                with col2:
                    st.write("**Standard Deviation (RGB):**")
                    st.write(f"- Red: {stat.stddev[0]:.2f}")
                    st.write(f"- Green: {stat.stddev[1]:.2f}")
                    st.write(f"- Blue: {stat.stddev[2]:.2f}")
                    
                    # Color variance
                    avg_stddev = sum(stat.stddev[:3]) / 3
                    st.metric("Color Variance", f"{avg_stddev:.2f}")
                
                # Dominant Colors
                st.subheader("🌈 Dominant Colors")
                
                # Resize for faster processing
                small_image = rgb_image.resize((150, 150))
                pixels = np.array(small_image).reshape(-1, 3)
                
                # Get most common colors
                pixel_tuples = [tuple(pixel) for pixel in pixels]
                color_counts = Counter(pixel_tuples)
                dominant_colors = color_counts.most_common(5)
                
                cols = st.columns(5)
                for idx, (color, count) in enumerate(dominant_colors):
                    with cols[idx]:
                        # Create color swatch
                        color_swatch = np.full((100, 100, 3), color, dtype=np.uint8)
                        st.image(color_swatch, caption=f"RGB{color}", use_container_width=True)
                        percentage = (count / len(pixel_tuples)) * 100
                        st.write(f"{percentage:.1f}%")
                
                # Image Type Classification
                st.subheader("🔍 Image Characteristics")
                
                # Determine if grayscale
                is_grayscale = image.mode in ['L', '1'] or (image.mode == 'RGB' and stat.stddev[0] < 10 and stat.stddev[1] < 10 and stat.stddev[2] < 10)
                
                # Determine brightness level
                if brightness < 85:
                    brightness_level = "Dark"
                elif brightness < 170:
                    brightness_level = "Medium"
                else:
                    brightness_level = "Bright"
                
                # Determine color richness
                if avg_stddev < 30:
                    color_richness = "Low (Monotone)"
                elif avg_stddev < 60:
                    color_richness = "Medium"
                else:
                    color_richness = "High (Vibrant)"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Type:** {'Grayscale' if is_grayscale else 'Color'}")
                with col2:
                    st.write(f"**Brightness:** {brightness_level}")
                with col3:
                    st.write(f"**Color Richness:** {color_richness}")
                
                st.success("✅ Image analysis complete!")

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

# ------------------ TEXT ANALYSIS TAB ------------------
with tab3:
    st.header("📝 Text Analysis")
    
    # Text input options
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload Text File"])
    
    text_to_analyze = ""
    
    if input_method == "Type/Paste Text":
        text_to_analyze = st.text_area("Enter text to analyze:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt", "md", "csv"])
        if uploaded_file:
            text_to_analyze = uploaded_file.read().decode("utf-8", errors='ignore')
            st.text_area("File Content:", text_to_analyze, height=200)
    
    if st.button("Analyze Text") and text_to_analyze.strip():
        with st.spinner("Analyzing text..."):
            
            # Basic Statistics
            st.subheader("📊 Basic Statistics")
            
            # Character count
            char_count = len(text_to_analyze)
            char_no_spaces = len(text_to_analyze.replace(" ", "").replace("\n", "").replace("\t", ""))
            
            # Word count
            words = text_to_analyze.split()
            word_count = len(words)
            
            # Sentence count
            sentences = sent_tokenize(text_to_analyze)
            sentence_count = len(sentences)
            
            # Paragraph count
            paragraphs = [p for p in text_to_analyze.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", char_count)
                st.metric("Characters (no spaces)", char_no_spaces)
            with col2:
                st.metric("Words", word_count)
                st.metric("Unique Words", len(set(words)))
            with col3:
                st.metric("Sentences", sentence_count)
                st.metric("Paragraphs", paragraph_count)
            with col4:
                avg_word_length = char_no_spaces / word_count if word_count > 0 else 0
                avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
                st.metric("Avg Word Length", f"{avg_word_length:.2f}")
                st.metric("Avg Words/Sentence", f"{avg_sentence_length:.2f}")
            
            # Readability Analysis
            st.subheader("📖 Readability Analysis")
            
            # Calculate reading time (average 200 words per minute)
            reading_time = word_count / 200
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Estimated Reading Time", f"{reading_time:.2f} minutes")
                
                # Simple complexity score based on average word and sentence length
                if avg_word_length < 4.5 and avg_sentence_length < 15:
                    complexity = "Easy"
                elif avg_word_length < 5.5 and avg_sentence_length < 20:
                    complexity = "Medium"
                else:
                    complexity = "Complex"
                st.write(f"**Complexity Level:** {complexity}")
            
            with col2:
                # Calculate lexical diversity (unique words / total words)
                lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
                st.metric("Lexical Diversity", f"{lexical_diversity:.2%}")
                st.caption("(Higher = more varied vocabulary)")
            
            # Sentiment Analysis
            st.subheader("😊 Sentiment Analysis")
            
            blob = TextBlob(text_to_analyze)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment classification
                if polarity > 0.1:
                    sentiment = "Positive 😊"
                    sentiment_color = "green"
                elif polarity < -0.1:
                    sentiment = "Negative 😞"
                    sentiment_color = "red"
                else:
                    sentiment = "Neutral 😐"
                    sentiment_color = "gray"
                
                st.write(f"**Overall Sentiment:** :{sentiment_color}[{sentiment}]")
                st.metric("Polarity Score", f"{polarity:.3f}")
                st.caption("(-1 = Very Negative, 0 = Neutral, +1 = Very Positive)")
            
            with col2:
                subjectivity_label = "Subjective" if subjectivity > 0.5 else "Objective"
                st.write(f"**Text Type:** {subjectivity_label}")
                st.metric("Subjectivity Score", f"{subjectivity:.3f}")
                st.caption("(0 = Objective, 1 = Subjective)")
            
            # Word Frequency Analysis
            st.subheader("🔤 Most Common Words")
            
            # Tokenize and clean
            words_lower = [word.lower() for word in words]
            
            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words_lower if word.isalnum() and word not in stop_words and len(word) > 2]
            
            if filtered_words:
                word_freq = Counter(filtered_words)
                top_words = word_freq.most_common(10)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create a simple bar chart
                    import pandas as pd
                    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    st.bar_chart(df.set_index('Word'))
                
                with col2:
                    st.write("**Top 10 Words:**")
                    for word, freq in top_words:
                        st.write(f"- **{word}**: {freq}")
            
            # Character Distribution
            st.subheader("🔢 Character Distribution")
            
            letter_count = sum(c.isalpha() for c in text_to_analyze)
            digit_count = sum(c.isdigit() for c in text_to_analyze)
            space_count = sum(c.isspace() for c in text_to_analyze)
            punct_count = sum(c in '.,;:!?"\'()[]{}' for c in text_to_analyze)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Letters", letter_count)
            with col2:
                st.metric("Digits", digit_count)
            with col3:
                st.metric("Spaces", space_count)
            with col4:
                st.metric("Punctuation", punct_count)
            
            # Language Detection
            st.subheader("🌍 Language Detection")
            try:
                detected_lang = blob.detect_language()
                lang_names = {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian',
                    'zh-CN': 'Chinese (Simplified)', 'ja': 'Japanese', 'ko': 'Korean',
                    'ar': 'Arabic', 'hi': 'Hindi'
                }
                lang_name = lang_names.get(detected_lang, detected_lang.upper())
                st.write(f"**Detected Language:** {lang_name}")
            except:
                st.write("**Detected Language:** Unable to detect (might be too short or mixed)")
            
            st.success("✅ Text analysis complete!")
    
    elif st.button("Analyze Text") and not text_to_analyze.strip():
        st.warning("Please enter or upload some text to analyze.")
