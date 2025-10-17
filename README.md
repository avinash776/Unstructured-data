# Unstructured Data Analysis - Setup & Run Guide

## Overview
Streamlit web app for image, audio, and text analysis with features like face detection, age/gender/emotion analysis, background removal, speech-to-text, text-to-speech, and comprehensive text analytics.

## Features
- **Image Analysis**: Basic stats, face detection, age/gender/emotion, background removal
- **Audio Analysis**: Text-to-speech, speech-to-text transcription
- **Text Analysis**: Statistics, sentiment analysis, readability metrics, word frequency

## Windows Setup

### Prerequisites
- Python 3.10+ installed
- Internet connection (for first-time model downloads)

### Quick Start
1. **Install dependencies**:
   ```cmd
   "C:\Users\addan\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r c:\ICT\Audio\requirements.txt
   ```

2. **Run the app**:
   ```cmd
   "C:\Users\addan\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run c:\ICT\Audio\app.py
   ```

3. **Open in browser**: http://localhost:8501

### Optional: Enable Audio Features
For full audio support, install ffmpeg:
```cmd
choco install ffmpeg -y
```
Then restart the app.

### Testing
Run the sanity check to verify dependencies:
```cmd
"C:\Users\addan\AppData\Local\Programs\Python\Python310\python.exe" c:\ICT\Audio\sanity_check.py
```

## Usage
1. **Image Analysis**: Upload JPG/PNG/etc., click analysis buttons
2. **Audio Analysis**: Type text for TTS, upload audio files for transcription
3. **Text Analysis**: Type/paste text or upload files for comprehensive analysis

## Troubleshooting
- First runs download AI models (~100MB total) - be patient
- Face detection requires clear frontal faces
- Background removal works best with distinct subjects
- If Streamlit errors, restart the app

### Streamlit Cloud deployment tips
- Add `runtime.txt` with `python-3.11.9` to pin a supported runtime.
- Use `numpy==1.26.4` to ensure wheels are available and avoid building from source on Python 3.13.
- Include `setuptools` and `wheel` in `requirements.txt` to prevent sdist build issues.
- If you hit `ModuleNotFoundError: No module named 'distutils'` during pip install, it typically means the environment is Python 3.13; pin to 3.11 via `runtime.txt`.

## File Structure
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `packages.txt` - System packages (for deployment)
- `sanity_check.py` - Dependency validation script