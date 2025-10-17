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
# Handle both old and new NLTK versions
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download punkt tokenizer (try new version first, then fall back to old)
try:
    from nltk.tokenize import sent_tokenize
    sent_tokenize("test")  # Test if it works
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

# Download stopwords
try:
    from nltk.corpus import stopwords
    stopwords.words('english')  # Test if it works
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ------------------ IMAGE ANALYSIS TAB ------------------
with tab1:
    st.header("🖼️ Complete Image Analysis")
    st.write("Upload an image and get comprehensive analysis including basic stats, face detection, object detection, age/gender/emotion prediction, and more!")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"], key="main_image_upload")
    
    if uploaded_image:
        # Load and display image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # Convert PIL to numpy for OpenCV
        image_np = np.array(rgb_image)
        
        # Single button to analyze everything
        if st.button("🔍 Analyze Everything", key="analyze_all", type="primary"):
            
            # ==========================
            # 1. BASIC ANALYSIS
            # ==========================
            with st.spinner("📊 Analyzing basic characteristics..."):
                st.markdown("---")
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
            
            # ==========================
            # 2. FACE DETECTION
            # ==========================
            with st.spinner("👤 Detecting faces..."):
                st.markdown("---")
                st.subheader("👤 Face Detection")
                try:
                    import cv2
                    
                    # Convert to OpenCV format
                    img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    
                    # Load Haar Cascade
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    # Draw rectangles
                    result_img = image_np.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    st.image(result_img, caption=f"Detected {len(faces)} Face(s)", use_container_width=True)
                    
                    if len(faces) > 0:
                        st.success(f"✅ Found {len(faces)} face(s) in the image!")
                        st.write("**Face Details:**")
                        for idx, (x, y, w, h) in enumerate(faces):
                            st.write(f"- Face {idx+1}: Position ({x}, {y}), Size {w}x{h} pixels")
                    else:
                        st.warning("No faces detected in the image.")
                
                except Exception as e:
                    st.error(f"Error during face detection: {str(e)}")
            
            # ==========================
            # 3. OBJECT DETECTION
            # ==========================
            with st.spinner("🎯 Detecting objects..."):
                st.markdown("---")
                st.subheader("🎯 Object Detection with YOLO")
                try:
                    from ultralytics import YOLO
                    
                    # Load YOLO model (will download on first use)
                    model = YOLO('yolov8n.pt')  # Using nano model for speed
                    
                    # Run detection
                    results = model(image_np)
                    
                    # Get annotated image
                    annotated_img = results[0].plot()
                    
                    st.image(annotated_img, caption="Objects Detected", use_container_width=True)
                    
                    # Display detection details
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        st.success(f"✅ Detected {len(boxes)} object(s)!")
                        
                        st.write("**Detected Objects:**")
                        detections = {}
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = model.names[cls]
                            
                            if label not in detections:
                                detections[label] = []
                            detections[label].append(conf)
                        
                        for obj, confs in detections.items():
                            avg_conf = sum(confs) / len(confs)
                            st.write(f"- **{obj}**: {len(confs)} instance(s) (avg confidence: {avg_conf:.2%})")
                    else:
                        st.info("No objects detected with high confidence.")
                
                except Exception as e:
                    st.error(f"Error during object detection: {str(e)}")
                    st.info("Note: First run will download the YOLO model (~6MB)")
            
            # ==========================
            # 4. AGE, GENDER, EMOTION
            # ==========================
            with st.spinner("🎭 Analyzing age, gender & emotion..."):
                st.markdown("---")
                st.subheader("🎭 Age, Gender & Emotion Analysis")
                try:
                    from deepface import DeepFace
                    
                    # Save image temporarily for DeepFace
                    temp_path = "temp_image.jpg"
                    rgb_image.save(temp_path)
                    
                    # Analyze image
                    analysis = DeepFace.analyze(temp_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Age", f"{analysis['age']} years")
                    
                    with col2:
                        gender = analysis['dominant_gender']
                        gender_conf = analysis['gender'][gender]
                        st.metric("Predicted Gender", gender.capitalize())
                        st.caption(f"Confidence: {gender_conf:.1f}%")
                    
                    with col3:
                        emotion = analysis['dominant_emotion']
                        emotion_conf = analysis['emotion'][emotion]
                        emotion_emoji = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 
                            'surprise': '😲', 'fear': '😨', 'disgust': '🤢', 
                            'neutral': '😐'
                        }
                        st.metric("Predicted Emotion", f"{emotion_emoji.get(emotion, '😐')} {emotion.capitalize()}")
                        st.caption(f"Confidence: {emotion_conf:.1f}%")
                    
                    # Show all emotion probabilities
                    st.write("**All Emotion Probabilities:**")
                    emotion_df = {k: f"{v:.1f}%" for k, v in analysis['emotion'].items()}
                    st.json(emotion_df)
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Note: Make sure there's a clear face in the image. First run will download models.")
            
            # ==========================
            # 5. BACKGROUND REMOVAL
            # ==========================
            with st.spinner("🖼️ Removing background..."):
                st.markdown("---")
                st.subheader("🖼️ Background Removal")
                try:
                    from rembg import remove
                    
                    # Remove background
                    output_img = remove(image_np)
                    
                    st.image(output_img, caption="Background Removed", use_container_width=True)
                    
                    # Save to session state for potential reuse
                    st.session_state['no_bg_image'] = output_img
                
                except Exception as e:
                    st.error(f"Error removing background: {str(e)}")
                    st.info("Note: First run will download the background removal model")
            
            # Final success message
            st.markdown("---")
            st.success("🎉 **Complete Image Analysis Finished!** All features have been analyzed.")
            
            # Optional: Background replacement section
            st.markdown("---")
            st.subheader("🔄 Optional: Replace Background")
            st.write("Want to replace the background with a custom image? Upload one below:")
            
            background_image = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"], key="bg_upload")
            
            if background_image and 'no_bg_image' in st.session_state:
                if st.button("Replace Background", key="replace_bg"):
                    try:
                        with st.spinner("Replacing background..."):
                            # Load background
                            bg_img = Image.open(background_image).convert('RGBA')
                            no_bg = Image.fromarray(st.session_state['no_bg_image'])
                            
                            # Resize background to match foreground
                            bg_img = bg_img.resize(no_bg.size, Image.Resampling.LANCZOS)
                            
                            # Composite images
                            combined = Image.alpha_composite(bg_img, no_bg)
                            
                            st.image(combined, caption="Background Replaced", use_container_width=True)
                            st.success("✅ Background replaced successfully!")
                    
                    except Exception as e:
                        st.error(f"Error replacing background: {str(e)}")
            elif not background_image:
                st.info("💡 Upload a background image above to replace the removed background")

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
    
    if st.button("Analyze Text"):
        if text_to_analyze.strip():
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
                try:
                    sentences = sent_tokenize(text_to_analyze)
                    sentence_count = len(sentences)
                except:
                    # Fallback: simple sentence splitting
                    sentences = [s.strip() for s in text_to_analyze.replace('!', '.').replace('?', '.').split('.') if s.strip()]
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
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    # Fallback: common English stopwords
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                                 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
                                 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
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
        else:
            st.warning("Please enter or upload some text to analyze.")

