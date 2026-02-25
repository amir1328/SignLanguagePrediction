# 🖐 Two-Way Sign Language Communicator (LSTM Deep Learning)

This is a real-time, two-way sign language recognition application built with Python, OpenCV, MediaPipe, and TensorFlow. 

It tracks your hand gestures via your webcam and uses a custom-trained **Long Short-Term Memory (LSTM) Neural Network** to analyze dynamic hand movements over time, predicting what sign you are making and translating it instantly.

For two-way communication, it features **Speech-to-Text (STT)** to listen to the hearing person, translates their words into animated Sign Language GIF avatars, and provides a WhatsApp-style chat history for both participants.

---

## 🚀 Features
*   **Sequential Hand Tracking**: Uses Google's MediaPipe Holistic to extract 126 3D coordinates from both hands continuously.
*   **Deep Learning LSTM**: Analyzes 30 consecutive frames at a time in a rolling buffer, allowing it to understand complex motion trajectories and moving signs rather than static imagery.
*   **Two-Way Chat UI**: A fluid 60FPS PyQt5 Desktop Application that tracks conversations like a chat app.
*   **Voice Output (TTS)**: Automatically speaks the sign out loud when you make a gesture.
*   **Avatar Translations**: Listens to the hearing person's voice using highly sensitive STT, transcribes it, and automatically plays your animated matching `.gif` files on the screen.

---

## 🛠️ Setup & Installation

1. Make sure you have Python installed.
2. Install the exact dependencies from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: To enable GPU Acceleration, ensure you have NVIDIA CUDA and cuDNN installed on your system. The scripts are pre-configured to automatically request GPU Memory Growth from TensorFlow).*

---

## 📖 How to Use

The application is split into three simple steps. Run these scripts in order from your terminal:

### Step 1: Record Temporal Data & Avatars 📸
Teach the AI what your signs look like in motion, and record the GIF that will be used for the Avatar translation.
```bash
python 1_collect_data.py
```
*   The terminal will ask you to enter a **Label** (e.g., `hello`, `eat`, `drink`).
*   *(It will automatically generate and save an MP3 audio file & a looping GIF in the background).*
*   A camera window will open. Position your hands and press **`R`**.
*   It will rapidly record **15 unique video sequences** (where every sequence lasts 30 frames). This teaches the AI the entire smooth motion of your gesture.
*   Repeat this step for *at least two* different signs!

### Step 2: Compile the Neural Network 🧠
Once you've collected data for a few signs, it's time to build the Tensorflow LSTM Architecture.
```bash
python 2_train_model.py
```
*   This reads your optimized `.npy` sequence arrays from the `dataset` folder.
*   Builds a multi-layer Neural Network explicitly designed for time-series recognition.
*   Trains for 200 epochs and spits out a compiled **Keras deep learning file**.
*   Saves the built brain to a file called `sign_lstm_model.keras`.

### Step 3: Two-Way Communication App ✨
Now test your app in the real world!
```bash
python 3_app_ui.py
```
*   **Signer:** Show your hands to the camera. The app maintains a 30-frame rolling window buffer in real-time. When it predicts a sign with **>75% confidence**, a Green chat bubble will appear on the right, and the computer will speak the word out loud.
*   **Hearing:** Speak into your microphone. A Gray chat bubble will transcribe what you are saying on the left.
*   **Avatar:** If you say a word recorded in Step 1, your matching Avatar GIF will instantly queue up on the screen and play!

---

## 📂 Project Structure
*   `1_collect_data.py` - Script to record temporal hand landmarks, TTS audio, and GIF avatars.
*   `2_train_model.py` - Script to train the TensorFlow LSTM model on your sequence arrays.
*   `3_app_ui.py` - The main PyQt5 Desktop Application (Vision, TTS, STT, and Inference UI).
*   `dataset/` - *(Auto-generated)* Folders containing the raw `.npy` arrays for each recorded sign.
*   `sign_lstm_model.keras` - *(Auto-generated)* The compiled Keras Deep Learning Model.
*   `actions.npy` - *(Auto-generated)* The indexing map bridging your string labels to Keras nodes.
*   `audio_signs/` - *(Auto-generated)* Directory where Google TTS MP3s are stored.
*   `sign_gifs/` - *(Auto-generated)* Directory where animated Avatar translations are stored.
