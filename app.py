import streamlit as st

# --- Streamlit compatibility shim ---
# Some environments may have older Streamlit versions where st.image uses
# 'use_column_width' instead of 'use_container_width'. This wrapper maps
# the newer kwarg to the older one automatically.
try:
    _orig_st_image = st.image
    def _st_image_compat(*args, **kwargs):
        try:
            return _orig_st_image(*args, **kwargs)
        except TypeError as e:
            if 'use_container_width' in kwargs:
                # map to legacy parameter and retry
                val = kwargs.pop('use_container_width')
                kwargs['use_column_width'] = val
                return _orig_st_image(*args, **kwargs)
            raise
    st.image = _st_image_compat
except Exception:
    # If anything goes wrong, keep default behavior
    pass
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

# ------------------ Environment diagnostic checks ------------------
# This block attempts to import critical packages used by the image analysis
# features and surfaces friendly, actionable guidance inside the Streamlit UI
# when imports fail (common on Streamlit Cloud when requirements.txt /
# packages.txt weren't applied or binaries are incompatible).
import importlib
import traceback

REQUIRED_MODULES = {
    'cv2': 'opencv-python-headless, opencv-contrib-python-headless',
    'deepface': 'deepface, tf-keras',
    'rembg': 'rembg, onnxruntime',
    'onnxruntime': 'onnxruntime',
}

def check_runtime_env():
    availability = {}
    errors = {}
    for mod, install_hint in REQUIRED_MODULES.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, '__version__', None)
            availability[mod] = ver
        except Exception as e:
            availability[mod] = None
            # Store only the error message, not full traceback to avoid crashes
            errors[mod] = str(e)
    return availability, errors

env_avail, env_errors = check_runtime_env()

try:
    # If running under Streamlit, surface diagnostics interactively
    import streamlit as _st_check
    if any(v is None for v in env_avail.values()):
        with _st_check.expander("🔧 Environment diagnostics (click to expand) — Missing/failing packages detected"):
            _st_check.write("Detected the following package issues. Update `requirements.txt` and `packages.txt` in your repo and redeploy on Streamlit Cloud.")
            for mod, err in env_errors.items():
                _st_check.write(f"• Module `{mod}`: failed to import.")
                _st_check.code(err if err else "No error details available")
                if mod == 'cv2':
                    _st_check.info("Recommendation: Ensure compatible numpy and OpenCV binaries. Example:\n\npip install numpy==1.24.3 opencv-python-headless==4.9.0.80 opencv-contrib-python-headless==4.9.0.80")
                
                elif mod == 'deepface':
                    _st_check.info("Recommendation: pip install deepface==0.0.87 tf-keras==2.15.0 tensorflow==2.15.0")
                elif mod in ('rembg', 'onnxruntime'):
                    _st_check.info("Recommendation: pip install rembg==2.0.55 onnxruntime==1.17.0")
                elif mod == 'tensorflow':
                    _st_check.info("Recommendation: pip install tensorflow==2.15.0 numpy==1.24.3")
            _st_check.write("After editing `requirements.txt`, commit & push to GitHub. Streamlit Cloud will rebuild the environment on the next deploy.")
except Exception:
    # non-streamlit runs or when streamlit import isn't available — ignore
    pass

st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ------------------ IMAGE ANALYSIS TAB ------------------
with tab1:
    st.header("🖼️ Image Analysis")
    st.write("Upload an image and choose which analysis to perform!")
    
    # Show package availability status
    missing = [m for m, v in env_avail.items() if v is None]
    if missing:
        st.warning(f"⚠️ Missing packages: {', '.join(missing)}. Some features may not work. Check Environment diagnostics above.")
    
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
        
        st.markdown("---")
        st.subheader("Choose Analysis Type:")
        
        # Create columns for buttons (object detection removed)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            basic_btn = st.button("📊 Basic Analysis", key="basic_analysis", use_container_width=True)
        with col2:
            face_btn = st.button("👤 Face Detection", key="face_detection", use_container_width=True)
        with col3:
            age_btn = st.button("🎭 Age/Gender/Emotion", key="age_analysis", use_container_width=True)
        with col4:
            bg_btn = st.button("🪄 Remove Background", key="bg_removal", use_container_width=True)
        
        st.markdown("---")
        
        # ==========================
        # 1. BASIC ANALYSIS
        # ==========================
        if basic_btn:
            with st.spinner("📊 Analyzing basic characteristics..."):
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
        if face_btn:
            with st.spinner("👤 Detecting faces..."):
                st.subheader("👤 Face Detection")
                try:
                    import os
                    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

                    # Try DeepFace (RetinaFace ➜ MTCNN) FIRST for higher accuracy
                    df_faces = []
                    used_backend = None
                    try:
                        from deepface import DeepFace
                        for backend in ['retinaface', 'mtcnn']:
                            try:
                                extracted = DeepFace.extract_faces(
                                    img_path=image_np,
                                    detector_backend=backend,
                                    enforce_detection=False
                                )
                                for f in extracted or []:
                                    area = f.get('facial_area') or f.get('region') or {}
                                    x, y = int(area.get('x', 0)), int(area.get('y', 0))
                                    w, h = int(area.get('w', 0)), int(area.get('h', 0))
                                    if w > 0 and h > 0:
                                        df_faces.append((x, y, w, h))
                                if df_faces:
                                    used_backend = backend
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass

                    faces = []
                    # If DeepFace failed to find faces, fallback to OpenCV Haar
                    if not df_faces:
                        import cv2
                        import urllib.request
                        # Prefer OpenCV's bundled haarcascade if available
                        cascade_path = None
                        try:
                            bundle_dir = getattr(cv2.data, 'haarcascades', None)
                            if bundle_dir:
                                candidate = os.path.join(bundle_dir, 'haarcascade_frontalface_default.xml')
                                if os.path.exists(candidate):
                                    cascade_path = candidate
                        except Exception:
                            cascade_path = None

                        if cascade_path is None:
                            local_path = 'haarcascade_frontalface_default.xml'
                            if not os.path.exists(local_path):
                                cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
                                try:
                                    urllib.request.urlretrieve(cascade_url, local_path)
                                    cascade_path = local_path
                                except Exception:
                                    cascade_path = None
                            else:
                                cascade_path = local_path

                        if not cascade_path or not os.path.exists(cascade_path):
                            raise RuntimeError('Haarcascade file not found or failed to download')

                        face_cascade = cv2.CascadeClassifier(cascade_path)
                        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

                        # Slightly more permissive minSize to catch smaller faces
                        min_side = min(image_np.shape[0], image_np.shape[1])
                        min_size = max(24, int(0.05 * min_side))  # at least 24px or 5% of shorter side
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.08,
                            minNeighbors=5,
                            minSize=(min_size, min_size)
                        )

                    total_faces = (len(df_faces) if df_faces else 0) + (len(faces) if isinstance(faces, (list, tuple, np.ndarray)) else 0)
                    if total_faces > 0:
                        st.success(f"✅ Detected {total_faces} face(s)!")
                        face_img = image_np.copy()
                        # Draw DeepFace results in orange
                        for (x, y, w, h) in df_faces:
                            import cv2 as _cv
                            _cv.rectangle(face_img, (x, y), (x+w, y+h), (255, 165, 0), 2)
                        # Draw Haar results in green
                        for (x, y, w, h) in faces:
                            import cv2 as _cv
                            _cv.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        caption_note = ""
                        if df_faces:
                            caption_note += f"DeepFace ({used_backend})"
                        if faces is not None and len(faces) > 0:
                            caption_note += (" • " if caption_note else "") + "OpenCV Haar"
                        if caption_note:
                            st.caption(f"Detectors: {caption_note}")
                        st.image(face_img, caption="Faces Detected", use_container_width=True)
                    else:
                        st.info("No faces detected. Try a clearer, frontal face with good lighting.")

                except ImportError as e:
                    st.error(f"❌ Required libraries not properly installed: {str(e)}")
                    if env_avail.get('deepface') is None:
                        st.info("💡 Fix: pip install deepface==0.0.87 tf-keras==2.15.0 tensorflow==2.15.0")
                    elif env_avail.get('cv2') is None:
                        st.info("💡 Fix: pip install numpy==1.24.3 opencv-python-headless==4.9.0.80 opencv-contrib-python-headless==4.9.0.80")
                except Exception as e:
                    st.error(f"❌ Error during face detection: {str(e)}")
                    if "libGL" in str(e):
                        st.info("💡 Missing system library. Ensure `packages.txt` contains: libgl1-mesa-glx")
                    else:
                        st.info("💡 First run may be downloading detectors. Please try again in a moment.")
        
        
        
        # ==========================
        # 4. AGE, GENDER, EMOTION
        # ==========================
        if age_btn:
            with st.spinner("🎭 Analyzing age, gender & emotion..."):
                st.subheader("🎭 Age, Gender & Emotion Analysis")
                try:
                    import os
                    from deepface import DeepFace
                    import numpy as _np
                    import tempfile as _tmp
                    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

                    # Detect and crop the largest face for higher accuracy
                    face_crop = None
                    extracted = []
                    try:
                        extracted = DeepFace.extract_faces(
                            img_path=image_np,
                            detector_backend='retinaface',
                            enforce_detection=False
                        ) or []
                    except Exception:
                        # fallback to MTCNN
                        try:
                            extracted = DeepFace.extract_faces(
                                img_path=image_np,
                                detector_backend='mtcnn',
                                enforce_detection=False
                            ) or []
                        except Exception:
                            extracted = []

                    if extracted:
                        # Choose the largest detected face
                        def _area(d):
                            a = d.get('facial_area') or d.get('region') or {}
                            return max(1, int(a.get('w', 0)) * int(a.get('h', 0)))
                        best = max(extracted, key=_area)
                        # DeepFace provides an aligned 224x224 face in 'face'
                        face_arr = best.get('face')
                        if isinstance(face_arr, np.ndarray):
                            face_crop = face_arr

                    def _to_bgr_uint8(arr: _np.ndarray) -> _np.ndarray:
                        a = arr
                        if a.dtype != _np.uint8:
                            a = a.astype('float32')
                            if a.max() <= 1.0:
                                a = (a * 255.0).clip(0, 255).astype('uint8')
                            else:
                                a = a.clip(0, 255).astype('uint8')
                        # if RGB, convert to BGR for DeepFace (OpenCV convention)
                        if a.ndim == 3 and a.shape[2] == 3:
                            a = a[:, :, ::-1].copy()
                        return a

                    analysis = None
                    if face_crop is not None:
                        # Use aligned face crop; ensure correct dtype/color
                        target_img = _to_bgr_uint8(face_crop)
                        analysis = DeepFace.analyze(
                            img_path=target_img,
                            actions=['age', 'gender', 'emotion'],
                            detector_backend='skip',
                            enforce_detection=False
                        )
                    else:
                        # Use temp file path to avoid RGB/BGR ambiguity
                        with _tmp.NamedTemporaryFile(suffix='.jpg', delete=False) as _f:
                            rgb_image.save(_f.name)
                            temp_path = _f.name
                        try:
                            analysis = DeepFace.analyze(
                                img_path=temp_path,
                                actions=['age', 'gender', 'emotion'],
                                detector_backend='retinaface',
                                enforce_detection=False
                            )
                        finally:
                            try:
                                if 'temp_path' in locals() and os.path.exists(temp_path):
                                    os.remove(temp_path)
                            except Exception:
                                pass

                    if isinstance(analysis, list) and analysis:
                        analysis = analysis[0]

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Predicted Age", f"{analysis['age']} years")

                    with col2:
                        gender = analysis['dominant_gender']
                        gender_conf = float(analysis['gender'][gender])
                        st.metric("Predicted Gender", gender.capitalize())
                        st.caption(f"Confidence: {gender_conf:.1f}%")

                    with col3:
                        emotion = analysis['dominant_emotion']
                        emotion_conf = float(analysis['emotion'][emotion])
                        emotion_emoji = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠',
                            'surprise': '😲', 'fear': '😨', 'disgust': '🤢',
                            'neutral': '😐'
                        }
                        st.metric("Predicted Emotion", f"{emotion_emoji.get(emotion, '😐')} {emotion.capitalize()}")
                        st.caption(f"Confidence: {emotion_conf:.1f}%")

                    # Show all emotion probabilities
                    st.write("**All Emotion Probabilities:**")
                    emotion_df = {k: f"{float(v):.1f}%" for k, v in analysis['emotion'].items()}
                    st.json(emotion_df)

                    if face_crop is not None:
                        st.caption("Analyzed cropped, aligned face for better accuracy.")

                except ImportError as e:
                    st.error(f"❌ DeepFace not properly installed: {str(e)}")
                    if env_avail.get('deepface') is None:
                        st.info("💡 Fix: Add `deepface==0.0.87 tf-keras==2.15.0 tensorflow==2.15.0` to requirements.txt and redeploy")
                    else:
                        st.info("💡 Run: `pip install deepface tf-keras`")
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    if "libGL" in str(e):
                        st.info("💡 Missing system library. Ensure `packages.txt` contains: libgl1-mesa-glx")
                    elif "No face" in str(e) or "Face could not be detected" in str(e):
                        st.warning("⚠️ No clear face detected in the image. Please use a photo with a visible frontal face.")
                    else:
                        st.info("💡 First run will download AI models (~100MB). This may take a few minutes.")
        
        # ==========================
        # 5. BACKGROUND REMOVAL
        # ==========================
        if bg_btn:
            with st.spinner("🖼️ Removing background..."):
                st.subheader("🖼️ Background Removal")
                try:
                    from rembg import remove
                    import os
                    import io as _io
                    
                    # Set environment variable
                    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                    
                    # Remove background
                    output_img = remove(image_np)
                    
                    # Normalize to PIL RGBA for consistent handling
                    no_bg_pil = None
                    if isinstance(output_img, (bytes, bytearray)):
                        try:
                            no_bg_pil = Image.open(_io.BytesIO(output_img)).convert('RGBA')
                        except Exception:
                            no_bg_pil = None
                    if no_bg_pil is None:
                        try:
                            no_bg_pil = Image.fromarray(output_img).convert('RGBA')
                        except Exception:
                            no_bg_pil = None
                    if no_bg_pil is None:
                        # As a fallback, just convert original to RGBA
                        no_bg_pil = rgb_image.convert('RGBA')
                    
                    st.image(no_bg_pil, caption="Background Removed", use_container_width=True)
                    
                    # Save to session state for potential reuse
                    st.session_state['no_bg_image_rgba'] = no_bg_pil
                
                except ImportError as e:
                    st.error(f"❌ Background removal library not properly installed: {str(e)}")
                    if env_avail.get('rembg') is None or env_avail.get('onnxruntime') is None:
                        st.info("💡 Fix: Add `rembg==2.0.55 onnxruntime==1.17.0` to requirements.txt and redeploy")
                    else:
                        st.info("💡 Run: `pip install rembg`")
                except Exception as e:
                    st.error(f"❌ Error removing background: {str(e)}")
                    if "libGL" in str(e):
                        st.info("💡 Missing system library. Ensure `packages.txt` contains: libgl1-mesa-glx")
                    else:
                        st.info("💡 First run will download the background removal model (~50MB). Check your internet connection.")

        # Optional: Background replacement section (persistent across reruns)
        if 'no_bg_image_rgba' in st.session_state:
            st.markdown("---")
            st.subheader("🔄 Optional: Replace Background")
            st.write("Use a custom image as the new background for the subject extracted above.")

            background_image = st.file_uploader(
                "Upload background image", type=["jpg", "jpeg", "png"], key="bg_upload"
            )

            if background_image and st.button("Replace Background", key="replace_bg"):
                try:
                    with st.spinner("Replacing background..."):
                        # Load background and ensure RGBA
                        bg_img = Image.open(background_image).convert('RGBA')
                        no_bg = st.session_state['no_bg_image_rgba']

                        # Resize background to match foreground
                        bg_img = bg_img.resize(no_bg.size, Image.Resampling.LANCZOS)

                        # Composite images: background underneath, foreground (with alpha) on top
                        combined = Image.alpha_composite(bg_img, no_bg)

                        # Persist and show
                        st.session_state['replaced_bg_image'] = combined
                        st.image(combined, caption="Background Replaced", use_container_width=True)
                        st.success("✅ Background replaced successfully!")

                        # Offer download
                        import io as _dl_io
                        buf = _dl_io.BytesIO()
                        combined.save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="⬇️ Download Result (PNG)",
                            data=buf,
                            file_name="background_replaced.png",
                            mime="image/png",
                            key="download_replaced_bg"
                        )
                except Exception as e:
                    st.error(f"Error replacing background: {str(e)}")

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

