# Sign Language Companion - Unified Workspace

A state-of-the-art, multi-modal sign language translation and communication suite. This application leverages Deep Learning (LSTM) and MediaPipe Holistic tracking to bridge the gap between sign language and spoken/written language in real-time.

## 🚀 Key Features

- **Sequential Hand Tracking** — Powered by MediaPipe Holistic, extracting 126 wrist-relative 3D landmarks for high-precision motion analysis.
- **Deep Learning LSTM** — A custom-trained Long Short-Term Memory neural network analyzes 30-frame rolling sequences to predict dynamic signs.
- **Unified PyQt5 Workspace** — One app to rule them all. Access Communicator, Video, Text, and Audio translators from a single high-performance interface.
- **Side-by-Side Dual Translation** — Compare **Hybrid** (real GIF/MP4 signs) and **AI Skeleton** animations side-by-side.
- **High-Fidelity AI Skeletons** — Clean, structured facial features and body connections for clear AI-generated sign demonstrations.
- **Universal Export** — Stitch sequences together and export your translations as MP4 videos.
- **Real-time Audio STT** — Live microphone listening and audio file upload support for instant speech-to-sign translation.

---

## 🛠️ Main Modules

### 1. **COMMUNICATOR**
Real-time webcam interface. Sign to the camera to see instant text bubbles and hear voice output (TTS). Speak to see matching sign avatars instantly.

### 2. **VIDEO TRANSLATE**
Upload any video file. The system transcribes the speech and converts it into a continuous sequence of sign language GIFs or AI skeletons.

### 3. **TEXT TRANSLATE (Dual Mode)**
Type or paste any sentence. Instantly view both real sign media and AI-generated skeleton animations side-by-side for every word.

### 4. **AUDIO TRANSLATE**
Upload audio files or use live microphone input. Transcribes spoken words and displays them as sign language sequences. Supports MP4 export for both Hybrid and AI modes.

### 5. **DATASET BUILDER & TRAINER**
Record your own signs using the integrated AI recorder or fetch data from the **MS-ASL** cloud dataset. Train or optimize your LSTM model directly within the app.

---

## 💻 Setup & Installation

1. **Prerequisites:** Python 3.9–3.11.
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install yt-dlp pose-format imageio[ffmpeg]
   ```
3. **Launch the Workspace:**
   ```bash
   python SignLangCompanion.py
   ```

---

## 📂 Project Structure

| File | Purpose |
|---|---|
| `SignLangCompanion.py` | The main unified application hub. |
| `1_collect_data.py` | Real-time data collection tool for new signs. |
| `2_train_model.py` | Training script for the LSTM neural network. |
| `3_app_ui.py` | The core Communicator widget logic. |
| `4_video_translate.py` | Video file translation and AI skeleton renderer. |
| `5_text_to_sign.py` | Text-to-Sign translation with Dual Sign Mode. |
| `8_audio_translate.py` | Audio file and mic-based sign translator. |
| `asl_gloss.py` | English-to-ASL linguistic processing engine. |
| `sign_gifs/` | Local library of sign language media. |
| `ai_signs/` | Cache for high-quality AI-generated skeleton videos. |

---

## 🌟 Enhanced AI Skeleton
The AI renderer has been optimized for clarity:
- **Clean Face Mesh:** Dotted facial features for expression without clutter.
- **Anatomical Accuracy:** Proper wrist-bridge connections and limb proportions.
- **Continuous Motion:** Relaxed confidence thresholds for fluid, natural signing.

Developed with ❤️ for accessibility.
