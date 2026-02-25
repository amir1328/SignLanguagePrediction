import sys
import os
import time
import queue
import threading
import pickle
import cv2
import mediapipe as mp

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
MODEL_FILE         = "sign_model.pkl"
PREDICTION_HISTORY = 7
TTS_COOLDOWN       = 1.5

# =============================================================================
# 1. Camera Handling (from 1_collect_data.py)
# =============================================================================
def get_camera():
    backends = [
        ("DirectShow (DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (MSMF)", cv2.CAP_MSMF),
        ("Default (ANY)", cv2.CAP_ANY)
    ]
    indices = [1, 0, 2]
    
    for index in indices:
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                else:
                    cap.release()
    return None

# =============================================================================
# 2. Qt Threads for Hardware & AI
# =============================================================================

class CameraThread(QThread):
    """Processes OpenCV and MediaPipe data on a background thread to keep UI fast."""
    change_pixmap_signal = pyqtSignal(QImage)
    sign_detected_signal = pyqtSignal(str)
    
    def __init__(self, model):
        super().__init__()
        self._run_flag = True
        self.model = model
        
    def run(self):
        cap = get_camera()
        if not cap:
            return
            
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils
        hands    = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        history = []
        last_spoken = ""
        last_spoken_time = 0.0

        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1)
            h, w, ch = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            current_sign = ""
            
            # Machine Learning Prediction
            if result.multi_hand_landmarks:
                features = []
                for i in range(min(2, len(result.multi_hand_landmarks))):
                    hand_lms = result.multi_hand_landmarks[i]
                    mp_draw.draw_landmarks(
                        rgb_frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )
                    for lm in hand_lms.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                        
                while len(features) < 126:
                    features.append(0.0)
                    
                if self.model:
                    pred = self.model.predict([features])[0]
                    history.append(pred)
                    if len(history) > PREDICTION_HISTORY:
                        history.pop(0)
                    current_sign = max(set(history), key=history.count)
            else:
                history.append("")
                if len(history) > PREDICTION_HISTORY:
                    history.pop(0)
                if history:
                    current_sign = max(set(history), key=history.count)
                    
            # Logic for when to trigger a new chat bubble / TTS
            now = time.time()
            if current_sign and current_sign != "" and current_sign != last_spoken and (now - last_spoken_time) > TTS_COOLDOWN:
                self.sign_detected_signal.emit(current_sign)
                last_spoken = current_sign
                last_spoken_time = now
            elif current_sign == "":
                last_spoken = ""
            
            # Convert frame to Qt Format and emit
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
        
    def speak(self, text):
        self.tts_q.put(text)
        
    def run(self):
        # 1. Start STT loop in a secondary demon thread so we don't block the TTS loop
        threading.Thread(target=self.stt_loop, daemon=True).start()
        
        # 2. Main TTS Loop
        while self._run_flag:
            try:
                # Wait for something in the queue with a timeout so we can exit cleanly
                text = self.tts_q.get(timeout=0.5)
                if text == "QUIT_COMMAND":
                    break
                    
                audio_file = os.path.join("audio_signs", f"{text.lower()}.mp3")
                if not os.path.exists(audio_file):
                    # fallback dynamic generation if file missing
                    tts = gtts.gTTS(text=text, lang='en')
                    audio_file = f"temp_{int(time.time())}.mp3"
                    tts.save(audio_file)
                    
                pygame.mixer.init()
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
                print(f"[Audio Error] {e}")

    def stt_loop(self):
        recognizer = sr.Recognizer()
        # Tighten the limits so it translates sentences much faster
        recognizer.pause_threshold = 0.5
        recognizer.dynamic_energy_threshold = True
        
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            while self._run_flag:
                try:
                    # Timeout = time it waits for someone to START speaking
                    # phrase_time_limit = maximum length of a sentence before it cuts and translates
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=7)
                    text = recognizer.recognize_google(audio)
                    if text:
                        self.heard_signal.emit(text)
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    pass
                    
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
        self.setWindowTitle("Two-Way Sign Language Communicator")
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
        self.status_label = QLabel("Initializing Systems...")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #aaaaaa;")
        left_layout.addWidget(self.status_label)
        
        main_layout.addWidget(left_panel, stretch=6)
        
        # --- RIGHT PANEL (Chat Bubbles) ---
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #2b2b2b; border-radius: 15px;")
        right_layout = QVBoxLayout(right_panel)
        
        header = QLabel("Chat History")
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
        self.add_bubble("System Ready. Start signing or speaking to begin.", sender="system")

    def init_ai(self):
        # Load Model
        model = None
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
                
        # Start Threads
        self.audio_thread = AudioWorker()
        self.audio_thread.heard_signal.connect(self.on_speech_heard)
        self.audio_thread.start()
        
        self.camera_thread = CameraThread(model)
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.sign_detected_signal.connect(self.on_sign_detected)
        self.camera_thread.start()
        self.status_label.setText("🟢 Live: Camera and Microphone Active")

    def update_image(self, qt_image):
        """Updates the video label with the latest OpenCV frame."""
        scaled_img = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_img))

    def add_bubble(self, text, sender):
        """Adds a WhatsApp style chat bubble to the scroll view."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 5, 0, 5)
        
        msg_label = QLabel(text)
        msg_label.setFont(QFont("Segoe UI", 12))
        msg_label.setWordWrap(True)
        msg_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        
        if sender == "signer":
            # Right-aligned, green
            msg_label.setStyleSheet("background-color: #00882b; color: white; padding: 12px; border-radius: 12px;")
            row_layout.addStretch()
            row_layout.addWidget(msg_label)
        elif sender == "hearing":
            # Left-aligned, dark gray
            msg_label.setStyleSheet("background-color: #4a4a4a; color: white; padding: 12px; border-radius: 12px;")
            row_layout.addWidget(msg_label)
            row_layout.addStretch()
        else:
            # System text
            msg_label.setStyleSheet("color: #888888; font-style: italic;")
            msg_label.setAlignment(Qt.AlignCenter)
            row_layout.addWidget(msg_label)
            
        self.chat_layout.addWidget(row)
        
        # Auto-scroll to bottom
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def on_sign_detected(self, sign_text):
        """Triggered when the signer completes a valid sign structure."""
        self.add_bubble(sign_text.capitalize(), sender="signer")
        self.audio_thread.speak(sign_text) # trigger TTS 

    def on_speech_heard(self, text):
        """Triggered when the hearing person speaks into the microphone."""
        self.add_bubble(text, sender="hearing")
        
        # Check if any detected words have matching GIFs
        words = text.lower().split()
        for word in words:
            # Strip punctuation like "hi!" to "hi"
            clean_word = ''.join(e for e in word if e.isalnum())
            gif_path = os.path.join("sign_gifs", f"{clean_word}.gif")
            if os.path.exists(gif_path):
                self.play_avatar_gif(gif_path)

    def play_avatar_gif(self, path):
        """Adds a new GIF to the queue, shifting older ones to the right."""
        # Shift the paths array instead of QMovies
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
                
        # Force PyQt to immediately redraw the screen so 
        # multiple rapid STT detections visually update all labels
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
