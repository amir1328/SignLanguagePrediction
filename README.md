# Two-Way Sign Language Communicator (LSTM Deep Learning)

A real-time, two-way sign language recognition application built with Python, OpenCV, MediaPipe, and TensorFlow.

It tracks hand gestures via webcam and uses a custom-trained **Long Short-Term Memory (LSTM) Neural Network** to analyze dynamic hand movements over time, predicting what sign you are making and translating it instantly.

For two-way communication, it features **Speech-to-Text (STT)** that listens to the hearing person, translates their words into animated Sign Language GIF avatars, and provides a WhatsApp-style chat history for both participants.

---

## Features

- **Sequential Hand Tracking** — MediaPipe Holistic extracts 126 wrist-relative 3D coordinates from both hands continuously.
- **Deep Learning LSTM** — Analyzes 30 consecutive frames in a rolling buffer to understand motion trajectories.
- **Two-Way Chat UI** — 60 FPS PyQt5 desktop application with a chat-style conversation history.
- **Voice Output (TTS)** — Automatically speaks the detected sign out loud.
- **Avatar Translations** — Listens to the hearing person via microphone, transcribes speech, and plays matching animated GIFs.
- **Video Translator** — Upload any video, extract its audio, transcribe it, and display the matching sign GIF for each word.
- **Text to Sign** — Type or paste any text and instantly see the sign GIF for each word.
- **MS-ASL Dataset Downloader** — Automatically download and convert clips from the Microsoft MS-ASL dataset to augment your training data.

---

## Setup & Installation

1. Make sure Python 3.9–3.11 is installed.
2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   pip install yt-dlp
   ```

> To enable GPU acceleration, install NVIDIA CUDA and cuDNN. TensorFlow is pre-configured to use Memory Growth automatically.

---

## How to Use

### Step 1 — Record Signs (`1_collect_data.py`)
Teach the model what your signs look like in motion and create the GIF avatar.

```bash
python 1_collect_data.py
```

- Enter a label (e.g. `hello`, `help`, `water`).
- Press `R` in the camera window to begin recording.
- Records **50 sequences × 30 frames** with wrist-relative normalization.
- Automatically generates a looping `.gif` and a TTS `.mp3` for each sign.
- Repeat for every sign you want to recognize.

---

### Step 2 — Train the Model (`2_train_model.py`)

```bash
python 2_train_model.py
```

- Reads `.npy` arrays from `dataset/`.
- Trains a multi-layer LSTM for 200 epochs with dropout and data augmentation.
- Saves `sign_lstm_model.keras` and `actions.npy`.

---

### Step 3 — Run the App (`3_app_ui.py`)

```bash
python 3_app_ui.py
```

- **Signer:** Show your hands to the camera. When a sign is predicted with >75% confidence, a green bubble appears and the word is spoken aloud.
- **Hearing:** Speak into the microphone. A gray bubble transcribes what you say.
- **Avatar:** If a word matches a recorded sign, the GIF plays on screen.

---

### Step 4 — Video Translator (`4_video_translate.py`)

Upload a video file, extract and transcribe its audio, then display a sign GIF card for each recognized word.

```bash
python 4_video_translate.py
```

- Browse and preview any video (MP4, AVI, MOV, MKV).
- Click **Translate** — audio is extracted and sent to Google Speech-to-Text.
- Each word is matched to a GIF in `sign_gifs/` and shown as a card.
- Export the matched GIFs as a new video with **Export Video**.

---

### Step 5 — Text to Sign (`5_text_to_sign.py`)

Type or paste any text and instantly see the matching sign GIF for each word. No camera, no audio needed.

```bash
python 5_text_to_sign.py
```

- Supports multi-word phrases (e.g. "what's up" matched as one sign).
- Words with no recorded sign show a "no sign" card.
- Press **Ctrl+Enter** to translate instantly.

---

## Expanding the Sign Library

### Option A — Record your own signs
Run `1_collect_data.py` and record 50 sequences for each new sign.

### Option B — Convert existing videos (`6_video_to_dataset.py`)
Convert a folder of sign language videos into `.npy` training data.

```bash
python 6_video_to_dataset.py C:/path/to/input_videos
```

Input folder can be labeled subfolders or a flat folder with filename prefixes:
```
input_videos/
  hello/
    clip1.mp4
    clip2.mp4
  help/
    clip1.mp4
```

### Option C — MS-ASL Dataset (`7_msasl_downloader.py`)
Download clips from the Microsoft MS-ASL dataset (1,000 ASL signs from YouTube) and convert them automatically.

```bash
# Download specific words
python 7_msasl_downloader.py --words "hello,help,water,please"

# Download top 20 most common signs
python 7_msasl_downloader.py --top 20
```

Requires the MS-ASL JSON files in `C:\Users\<you>\Downloads\MS-ASL\MS-ASL\`.
Downloads YouTube clips automatically (no ffmpeg required), trims to timestamps, runs through MediaPipe, and saves `.npy` files into `dataset/`.

> Note: The MS-ASL dataset links are from 2019. Expect ~30–50% of links to still be available.

After adding new data, retrain:
```bash
python 2_train_model.py
```

---

## Project Structure

| File / Folder | Description |
|---|---|
| `1_collect_data.py` | Record temporal hand landmarks, TTS audio, and GIF avatars |
| `2_train_model.py` | Train the TensorFlow LSTM model |
| `3_app_ui.py` | Main PyQt5 two-way communication app |
| `4_video_translate.py` | Upload a video and translate its speech to sign GIFs |
| `5_text_to_sign.py` | Type text and instantly view sign GIF cards |
| `6_video_to_dataset.py` | Convert video files to `.npy` training sequences |
| `7_msasl_downloader.py` | Download and convert MS-ASL dataset clips |
| `dataset/` | Auto-generated `.npy` sequence arrays per sign |
| `sign_gifs/` | Auto-generated animated GIF avatars |
| `audio_signs/` | Auto-generated TTS MP3 files |
| `sign_lstm_model.keras` | Trained Keras LSTM model |
| `actions.npy` | Label index map for the model |
| `requirements.txt` | All Python dependencies |
