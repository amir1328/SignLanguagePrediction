# 🖐 Sign Language Recognition App

This is a real-time sign language recognition application built with Python, OpenCV, and MediaPipe. 

It tracks your hand gestures via your webcam, uses a custom-trained AI model (`RandomForestClassifier`) to predict what sign you are making, and provides real-time **audio feedback (Text-to-Speech)** and **transcription (Speech-to-Text)**.

---

## 🚀 Features
*   **Real-time Hand Tracking**: Uses Google's MediaPipe to instantly extract 63 3D coordinates from your hand.
*   **Custom AI Training**: Easily train the model to recognize *your* specific signs in less than 5 seconds.
*   **Voice Output (TTS)**: Automatically speaks the sign out loud when you make a gesture (using Google TTS `gTTS` and `pygame`).
*   **Speech Transcription (STT)**: Listens to your voice in the background and transcribes it on the screen (using `SpeechRecognition`).
*   **Robust Camera Handling**: Built-in fallback to DirectShow to ensure it works flawlessly on Windows laptops, automatically bypassing virtual cameras.

---

## 🛠️ Setup & Installation

1. Make sure you have Python installed.
2. Install the exact dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 How to Use

The application is split into three simple steps to ensure you get the best accuracy. Run these scripts in order from your terminal:

### Step 1: Record Your Signs 📸
Use this script to teach the AI what your signs look like.
```bash
python 1_collect_data.py
```
*   The terminal will ask you to enter a **Label** (e.g., `hello`, `yes`, `peace`).
*   *(It will automatically generate and save an MP3 audio file for that word in the background).*
*   A camera window will open. Position your hand and press **`R`**.
*   It will rapidly take 500 photos of your hand coordinates and save them to `dataset.csv`.
*   Repeat this step for *at least two* different signs so the AI has something to compare!

### Step 2: Train the AI 🧠
Once you've collected data for a few signs, it's time to build the "Brain".
```bash
python 2_train_model.py
```
*   This reads your `dataset.csv` file.
*   Trains a Random Forest Machine Learning model.
*   Prints out an accuracy score (usually 99%-100%).
*   Saves the built brain to a file called `sign_model.pkl`.

### Step 3: Live Application (The Magic) ✨
Now test your app in the real world!
```bash
python 3_live_app.py
```
*   Show your hand to the camera; the **Top Green Bar** will show the translated sign.
*   The app will **Speak** the word out loud instantly.
*   Speak into your microphone; the **Bottom Orange Bar** will transcribe what you are saying.
*   Press **`ESC`** or **`Q`** to quit.

---

## 📂 Project Structure
*   `1_collect_data.py` - Script to record hand landmarks & automatically generate TTS audio.
*   `2_train_model.py` - Script to train the Scikit-Learn model on your data.
*   `3_live_app.py` - The main application combining Vision, TTS, and STT.
*   `app.py` - An alternative Web-based UI built using Streamlit.
*   `dataset.csv` - *(Auto-generated)* Raw coordinate data.
*   `sign_model.pkl` - *(Auto-generated)* The compiled AI Model.
*   `audio_signs/` - *(Auto-generated)* Directory where the pre-downloaded Google TTS MP3s are stored.
