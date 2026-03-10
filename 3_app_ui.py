import sys
import os
import time
import queue
import threading
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

import gtts
import pygame
import speech_recognition as sr

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QMovie, QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QFrame, QSizePolicy
)

# Configuration
MODEL_FILE         = "sign_lstm_model.keras"
TTS_COOLDOWN       = 1.5

# Enable GPU Memory Growth for TensorFlow to prevent UI freezing / VRAM crashing
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[!] Warning limiting GPU growth: {e}")

# No face indices needed anymore

def extract_keypoints(results):
    """Wrist-relative normalization — MUST match 1_collect_data.py exactly."""
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        lh = (lh - lh[0]).flatten()  # Subtract wrist (landmark 0)
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        rh = (rh - rh[0]).flatten()  # Subtract wrist (landmark 0)
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([lh, rh])

# =============================================================================
# 1. Camera Handling
# =============================================================================
def get_camera():
    backends = [("DirectShow (DSHOW)", cv2.CAP_DSHOW), ("Media Foundation (MSMF)", cv2.CAP_MSMF), ("Default (ANY)", cv2.CAP_ANY)]
    for index in [1, 0, 2]:
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
    return None

# =============================================================================
# 2. Qt Threads for Hardware & AI
# =============================================================================

class CameraThread(QThread):
    """Processes OpenCV and MediaPipe Holistic data on a background thread."""
    change_pixmap_signal = pyqtSignal(QImage)
    sign_detected_signal = pyqtSignal(str)
    
    def __init__(self, model_file):
        super().__init__()
        self._run_flag = True
        self.model_file = model_file
        
    def run(self):
        cap = get_camera()
        if not cap:
            return
            
        # Version-safe model loading:
        # .h5 is the recommended cross-Keras-version format.
        # .keras (native) is version-specific and may fail on different machines.
        model = None
        for candidate in ['sign_lstm_model.h5', 'sign_lstm_model.keras', self.model_file]:
            if not os.path.exists(candidate):
                continue
            try:
                model = tf.keras.models.load_model(candidate)
                print(f"[Model] Loaded from '{candidate}'")
                break
            except Exception as e:
                print(f"[Model] Could not load '{candidate}': {e}")

        if model is None:
            print("[Model] ERROR: Could not load any model file.")
            print("  → Keras version mismatch detected. Please re-run:")
            print("      python 2_train_model.py")
            print("  on THIS machine to generate a compatible model.")
            
        actions = np.load('actions.npy') if os.path.exists('actions.npy') else []
        
        mp_holistic = mp.solutions.holistic
        mp_draw  = mp.solutions.drawing_utils
        
        sequence = []
        predictions = [] # Keep a rolling log of recent predictions
        last_spoken = ""
        last_spoken_time = 0.0

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self._run_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                    
                frame = cv2.flip(frame, 1)
                h, w, ch = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Visuals
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # Machine Learning Prediction Vector
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:] # Rolling window of 30 frames
                
                current_sign = ""
                
                # Only predict if we have a full rolling window of 30 frames AND the model exists
                if len(sequence) == 30 and model is not None and len(actions) > 0:
                    # Keras expects (batches, timesteps, features)
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    # Log the index of what the AI thought
                    predictions.append(np.argmax(res))
                    
                    # We only analyze the last 10 predictions 
                    # This prevents rapid "flickering" between two similar signs mid-motion
                    if len(predictions) >= 10:
                        predictions = predictions[-10:]
                        
                        # Find the most frequently predicted class over the last 10 frames
                        majority_vote = np.bincount(predictions[-10:]).argmax()
                        
                        # CONFIDENCE THRESHOLD: 75% — appropriate for ~50 seqs per sign
                        # Raise to 0.85+ once you have 100+ seqs per sign
                        if res[majority_vote] > 0.75:
                            detected = actions[majority_vote]
                            # Suppress the idle/null class — it means "no sign"
                            if detected.lower() != "idle":
                                current_sign = detected
                    
                # Debounce/Cooldown Logic
                now = time.time()
                if current_sign and current_sign != last_spoken and (now - last_spoken_time) > TTS_COOLDOWN:
                    self.sign_detected_signal.emit(current_sign)
                    last_spoken = current_sign
                    last_spoken_time = now
                elif not current_sign:
                    # No confident sign detected — reset lock so same sign can fire again after a pause
                    last_spoken = ""

                # --- DEBUG OVERLAY: live confidence bars ---
                if len(sequence) == 30 and model is not None and len(actions) > 0:
                    bar_x, bar_y = 10, h - (len(actions) * 30) - 10
                    for i, action in enumerate(actions):
                        conf = float(res[i])
                        bar_w = int(conf * 200)
                        color = (0, 215, 100) if conf > 0.75 else (0, 150, 255)
                        y = bar_y + i * 30
                        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 20), color, -1)
                        cv2.rectangle(frame, (bar_x, y), (bar_x + 200, y + 20), (180, 180, 180), 1)
                        cv2.putText(frame, f"{action}: {conf:.0%}", (bar_x + 205, y + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # Convert the annotated BGR frame → RGB for Qt display
                # (rgb_frame was captured before landmarks/overlays were drawn, so use frame instead)
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
            
        cap.release()
        
    def stop(self):
        self._run_flag = False
        self.wait()


class AudioWorker(QThread):
    """Background STT listener + TTS queue player."""
    heard_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tts_q = queue.Queue()
        # Initialize mixer once here, not on every TTS call
        pygame.mixer.init()
        
    def speak(self, text):
        self.tts_q.put(text)
        
    def run(self):
        # 1. Start STT loop in a secondary demon thread so we don't block the TTS loop
        threading.Thread(target=self.stt_loop, daemon=True).start()
        
        # 2. Main TTS Loop
        while self._run_flag:
            try:
                text = self.tts_q.get(timeout=0.5)
                if text == "QUIT_COMMAND":
                    break
                    
                audio_file = os.path.join("audio_signs", f"{text.lower()}.mp3")
                if not os.path.exists(audio_file):
                    tts = gtts.gTTS(text=text, lang='en')
                    audio_file = f"temp_{int(time.time())}.mp3"
                    tts.save(audio_file)
                    
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy() and self._run_flag:
                    time.sleep(0.1)
                pygame.mixer.music.unload()
                
                if "temp_" in audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)
                    
            except queue.Empty:
                pass
            except Exception as e:
                pass

    def stt_loop(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 0.5
        recognizer.dynamic_energy_threshold = True
        
        try:
            mic = sr.Microphone()
        except Exception as e:
            print(f"\n[STT ERROR] Failed to access microphone: {e}")
            return
        with mic as source:
            print("\n[STT] Microphone configured. Calibrating noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            recognizer.energy_threshold = 300 # Lower threshold to catch softer words
            recognizer.dynamic_energy_threshold = False # Stop it from auto-muting you
            
            print("[STT] Listening in background...")
            while self._run_flag:
                try:
                    # Give it strict boundaries so it processes words instantly
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                    print("[STT] Processing audio...")
                    text = recognizer.recognize_google(audio)
                    if text:
                        print(f"[STT] Heard: {text}")
                        self.heard_signal.emit(text)
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"[STT ERROR] Recognition failed: {e}")
                    
    def stop(self):
        self._run_flag = False
        self.tts_q.put("QUIT_COMMAND")
        self.wait()

# =============================================================================
# 3. Main Qt Application Window
# =============================================================================

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Way Sign Language Communicator (LSTM GPU)")
        self.resize(1200, 750)
        self.setStyleSheet("background-color: #1a1a1a; color: white; font-family: Segoe UI, sans-serif;")
        
        # Core Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # --- LEFT PANEL (Camera & Avatar) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        
        left_layout.addWidget(self.video_label)
        
        # Avatar GIF Queue Layout (up to 3 recent signs)
        self.avatar_container = QWidget()
        self.avatar_layout = QHBoxLayout(self.avatar_container)
        self.avatar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.avatar_labels = []
        self.movies = [None, None, None]
        self.avatar_paths = [None, None, None]
        for _ in range(3):
            lbl = QLabel()
            lbl.setFixedSize(200, 150)
            lbl.setStyleSheet("background-color: transparent; border: 2px solid #00d7ff; border-radius: 8px;")
            lbl.setScaledContents(True)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.hide()
            self.avatar_layout.addWidget(lbl)
            self.avatar_labels.append(lbl)
            
        left_layout.addWidget(self.avatar_container, alignment=Qt.AlignCenter)
        
        # Status text below camera/avatar
        self.status_label = QLabel("Initializing Deep Learning Engine...")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #aaaaaa;")
        left_layout.addWidget(self.status_label)
        
        main_layout.addWidget(left_panel, stretch=6)
        
        # --- RIGHT PANEL (Chat Bubbles) ---
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 15px;")
        right_layout = QVBoxLayout(right_panel)
        
        header = QLabel("Communication History")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(header)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background-color: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.chat_container)
        
        right_layout.addWidget(self.scroll_area)
        main_layout.addWidget(right_panel, stretch=4)
        
        # --- Initialization ---
        self.init_ai()
        self.add_bubble("GPU LSTM Sequence Network Ready. Start signing or speaking.", sender="system")

    def init_ai(self):
        # Start Audio/STT Thread
        self.audio_thread = AudioWorker()
        self.audio_thread.heard_signal.connect(self.on_speech_heard)
        self.audio_thread.start()
        
        # Start Camera/Holistic Thread running Keras
        self.camera_thread = CameraThread(MODEL_FILE)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.sign_detected_signal.connect(self.on_sign_detected)
        self.camera_thread.start()
        
        self.status_label.setText("🟢 LSTM Network & Microphone Active")

    def update_image(self, qt_image):
        scaled_img = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_img))

    def add_bubble(self, text, sender):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 5, 0, 5)
        
        msg_label = QLabel(text)
        msg_label.setFont(QFont("Segoe UI", 12))
        msg_label.setWordWrap(True)
        msg_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
        if sender == "signer":
            msg_label.setStyleSheet("background-color: #00882b; color: white; padding: 12px; border-radius: 12px;")
            row_layout.addStretch()
            row_layout.addWidget(msg_label)
        elif sender == "hearing":
            msg_label.setStyleSheet("background-color: #4a4a4a; color: white; padding: 12px; border-radius: 12px;")
            row_layout.addWidget(msg_label)
            row_layout.addStretch()
        else:
            msg_label.setStyleSheet("color: #888888; font-style: italic;")
            msg_label.setAlignment(Qt.AlignCenter)
            row_layout.addWidget(msg_label)
            
        self.chat_layout.addWidget(row)
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def on_sign_detected(self, sign_text):
        self.add_bubble(sign_text.capitalize(), sender="signer")
        self.audio_thread.speak(sign_text)

    def on_speech_heard(self, text):
        self.add_bubble(text, sender="hearing")
        text_lower = text.lower()

        # 1. Try full phrase first — handles multi-word signs like "what's up"
        full_gif = os.path.join("sign_gifs", f"{text_lower}.gif")
        if os.path.exists(full_gif):
            self.play_avatar_gif(full_gif)
            return

        # 2. Fall back to individual words for single-word signs
        for word in text_lower.split():
            clean_word = ''.join(e for e in word if e.isalnum())
            gif_path = os.path.join("sign_gifs", f"{clean_word}.gif")
            if os.path.exists(gif_path):
                self.play_avatar_gif(gif_path)

    def play_avatar_gif(self, path):
        self.avatar_paths = [path, self.avatar_paths[0], self.avatar_paths[1]]
        for i in range(3):
            if self.movies[i]:
                self.movies[i].stop()
            p = self.avatar_paths[i]
            if p and os.path.exists(p):
                self.movies[i] = QMovie(p)
                self.avatar_labels[i].setMovie(self.movies[i])
                self.avatar_labels[i].show()
                self.movies[i].start()
            else:
                self.movies[i] = None
                self.avatar_labels[i].clear()
                self.avatar_labels[i].hide()
        QApplication.processEvents()
        
    def hide_all_avatar_gifs(self):
        self.avatar_paths = [None, None, None]
        for i in range(3):
            if self.movies[i]:
                self.movies[i].stop()
                self.movies[i] = None
            self.avatar_labels[i].clear()
            self.avatar_labels[i].hide()

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.audio_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
