# 🖐 Two-Way Sign Language Communicator

This is a real-time, two-way sign language recognition application built with Python, OpenCV, MediaPipe, and PyQt5. 

It tracks your hand gestures via your webcam, uses a custom-trained AI model (`RandomForestClassifier`) to predict what sign you are making, and translates it instantly. 

For two-way communication, it features **Speech-to-Text (STT)** to listen to the hearing person, translates their words into animated Sign Language GIF avatars, and provides a WhatsApp-style chat history for both participants.

---

## 🚀 Features
*   **Real-time Hand Tracking**: Uses Google's MediaPipe to instantly extract 126 3D coordinates (supporting two hands).
*   **Custom AI Training**: Easily train the model to recognize *your* specific signs in less than 5 seconds.
*   **Two-Way Chat UI**: A fluid 60FPS PyQt5 Desktop Application that tracks conversations like a chat app.
*   **Voice Output (TTS)**: Automatically speaks the sign out loud when you make a gesture.
*   **Avatar Translations**: Listens to the hearing person's voice using Google SpeechRecognition, transcribes it, and automatically plays your animated matching `.gif` files on the screen.

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

### Step 1: Record Your Signs & Avatars 📸
Use this script to teach the AI what your signs look like, and record the GIF that will be used for the Avatar translation.
```bash
python 1_collect_data.py
```
*   The terminal will ask you to enter a **Label** (e.g., `hello`, `yes`, `peace`).
*   *(It will automatically generate and save an MP3 audio file & a looping GIF in the background).*
*   A camera window will open. Position your hands and press **`R`**.
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

### Step 3: Two-Way Communication App ✨
Now test your app in the real world!
```bash
python 3_app_ui.py
```
*   **Signer:** Show your hands to the camera. When a sign is recognized, a Green chat bubble will appear on the right, and the computer will speak the word out loud.
*   **Hearing:** Speak into your microphone. A Gray chat bubble will transcribe what you are saying on the left.
*   **Avatar:** If the hearing person says a word you recorded in Step 1, your matching Avatar GIF will instantly pop up below the camera to translate it!

---

## 📂 Project Structure
*   `1_collect_data.py` - Script to record hand landmarks, TTS audio, and GIF avatars.
*   `2_train_model.py` - Script to train the Scikit-Learn model on your data.
*   `3_app_ui.py` - The main PyQt5 Desktop Application combining Vision, TTS, STT, and Chat UI.
*   `dataset.csv` - *(Auto-generated)* Raw coordinate data.
*   `sign_model.pkl` - *(Auto-generated)* The compiled AI Model.
*   `audio_signs/` - *(Auto-generated)* Directory where Google TTS MP3s are stored.
*   `sign_gifs/` - *(Auto-generated)* Directory where animated Avatar translations are stored.
