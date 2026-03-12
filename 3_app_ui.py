import sys
import os
import time
import json
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

try:
    from asl_gloss import english_to_asl, asl_gloss_string
except ImportError:
    def english_to_asl(text): return text.lower().split()
    def asl_gloss_string(text): return text.upper()

# Configuration
MODEL_FILE           = "sign_lstm_model.keras"
SIGN_ZONES_FILE      = "sign_zones.json"
TTS_COOLDOWN         = 1.5
CONFIDENCE_THRESHOLD = 0.85   # Raised from 0.75 for more reliable predictions
VOTE_WINDOW          = 15     # Majority vote over last N frames (was 10)
CONSEC_REQUIRED      = 5      # Consecutive same-sign frames required before emitting
MIN_HAND_FRAMES      = 15     # Min frames with real hand data in the 30-frame window
HAND_LOSS_RESET_SEC  = 1.0    # Reset buffer after this many seconds without a hand
ZONE_PENALTY         = 0.10   # Confidence penalty when hand is outside expected zone (reduced from 0.20)

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
        lh = (lh - lh[0]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        rh = (rh - rh[0]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([lh, rh])

def has_any_hand(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def get_wrist_pos(results):
    """Return absolute (x, y) wrist position of the most visible hand."""
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark[0]
        return (lm.x, lm.y)
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark[0]
        return (lm.x, lm.y)
    return None

def load_sign_zones():
    if os.path.exists(SIGN_ZONES_FILE):
        with open(SIGN_ZONES_FILE) as f:
            return json.load(f)
    return {}

def zone_confidence_multiplier(sign_name, wrist_pos, sign_zones):
    """
    Returns 1.0 if the wrist is inside the expected zone for this sign,
    or (1.0 - ZONE_PENALTY) if it's outside. No zone data → always 1.0.
    """
    if wrist_pos is None or sign_name not in sign_zones:
        return 1.0
    zone = sign_zones[sign_name]
    x_ok = zone['x'][0] <= wrist_pos[0] <= zone['x'][1]
    y_ok = zone['y'][0] <= wrist_pos[1] <= zone['y'][1]
    return 1.0 if (x_ok and y_ok) else (1.0 - ZONE_PENALTY)


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
# 0. Multimedia Helpers (MP4 / GIF Support)
# =============================================================================

class VideoPlayerThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_signal.emit(qt_img.copy())
            time.sleep(delay)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class AvatarView(QLabel):
    """A QLabel that can play either a QMovie (GIF) or a VideoPlayerThread (MP4)."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 150)
        self.setStyleSheet("background-color: #000; border: 2px solid #00d7ff; border-radius: 8px;")
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)
        self.movie = None
        self.vid_thread = None
        self.current_path = None
        self.hide()

    def play(self, path):
        self.stop()
        self.current_path = path
        self.show()

        if path.lower().endswith(('.mp4', '.avi', '.mov')):
            self.vid_thread = VideoPlayerThread(path)
            self.vid_thread.frame_signal.connect(self.update_frame)
            self.vid_thread.start()
        else:
            self.movie = QMovie(path)
            self.setMovie(self.movie)
            self.movie.start()

    def update_frame(self, qimg):
        self.setPixmap(QPixmap.fromImage(qimg))

    def stop(self):
        if self.movie:
            self.movie.stop()
            self.movie = None
        if self.vid_thread:
            self.vid_thread.stop()
            self.vid_thread = None
        self.clear()
        self.hide()
        self.current_path = None

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
        sign_zones = load_sign_zones()
        if sign_zones:
            print(f"[Zones] Loaded spatial zones for: {', '.join(sign_zones.keys())}")
        else:
            print("[Zones] No sign_zones.json found — spatial validation disabled.")

        mp_holistic = mp.solutions.holistic
        mp_draw     = mp.solutions.drawing_utils

        sequence          = []       # Rolling 30-frame window of keypoint vectors
        hand_frame_flags  = []       # Parallel bool list: True if frame had real hand data
        predictions       = []       # Rolling majority vote log
        consec_count      = 0        # Consecutive frames with same prediction above threshold
        last_candidate    = ""       # Sign being confirmed across consec_count frames
        last_spoken       = ""
        last_spoken_time  = 0.0
        last_hand_time    = time.time()  # Timestamp of last frame with a visible hand
        res               = None     # Last model output (for overlay)

        with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
            while self._run_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, ch = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = holistic.process(rgb_frame)

                # Draw hand landmarks
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                hand_visible = has_any_hand(results)
                wrist_pos    = get_wrist_pos(results)
                now          = time.time()

                # ── Gate 1: hand loss reset ────────────────────────────────────
                if hand_visible:
                    last_hand_time = now
                elif (now - last_hand_time) > HAND_LOSS_RESET_SEC:
                    # Hand has been gone > 1s — clear buffer and confirmation state
                    sequence         = []
                    hand_frame_flags = []
                    predictions      = []
                    consec_count     = 0
                    last_candidate   = ""
                    last_spoken      = ""

                # ── Accumulate keypoints ───────────────────────────────────────
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                hand_frame_flags.append(hand_visible)
                sequence          = sequence[-30:]
                hand_frame_flags  = hand_frame_flags[-30:]

                current_sign = ""

                # ── Gate 2: only predict if enough real hand data in window ────
                real_hand_count = sum(hand_frame_flags)
                can_predict = (
                    len(sequence) == 30
                    and model is not None
                    and len(actions) > 0
                    and real_hand_count >= MIN_HAND_FRAMES
                )

                if can_predict:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predictions.append(np.argmax(res))
                    predictions = predictions[-VOTE_WINDOW:]

                    if len(predictions) >= VOTE_WINDOW:
                        majority_idx  = np.bincount(predictions).argmax()
                        raw_conf      = float(res[majority_idx])
                        detected_sign = str(actions[majority_idx])

                        # ── Gate 3: spatial zone validation ───────────────────
                        zone_mult     = zone_confidence_multiplier(detected_sign, wrist_pos, sign_zones)
                        adj_conf      = raw_conf * zone_mult
                        in_zone       = (zone_mult == 1.0)

                        if adj_conf > CONFIDENCE_THRESHOLD and detected_sign.lower() != "idle":
                            # ── Gate 4: consecutive-frame confirmation ─────────
                            if detected_sign == last_candidate:
                                consec_count += 1
                            else:
                                last_candidate = detected_sign
                                consec_count   = 1

                            if consec_count >= CONSEC_REQUIRED:
                                current_sign = detected_sign
                        else:
                            consec_count   = 0
                            last_candidate = ""

                        # ── Confidence bar overlay ─────────────────────────────
                        bar_x = 10
                        bar_y = h - (len(actions) * 28) - 10
                        for i, action in enumerate(actions):
                            c = float(res[i]) * zone_confidence_multiplier(str(action), wrist_pos, sign_zones)
                            bw = int(c * 180)
                            color = (0, 200, 80) if c > CONFIDENCE_THRESHOLD else (0, 130, 255)
                            y = bar_y + i * 28
                            cv2.rectangle(frame, (bar_x, y), (bar_x + bw, y + 18), color, -1)
                            cv2.rectangle(frame, (bar_x, y), (bar_x + 180, y + 18), (160, 160, 160), 1)
                            cv2.putText(frame, f"{action}: {c:.0%}",
                                        (bar_x + 185, y + 13),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                        # ── Zone status indicator ──────────────────────────────
                        zone_text  = "ZONE OK" if in_zone else "OUT OF ZONE"
                        zone_color = (0, 200, 80) if in_zone else (0, 100, 255)
                        cv2.putText(frame, zone_text, (w - 160, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 2)

                # ── No hand overlay ────────────────────────────────────────────
                if not hand_visible:
                    cv2.putText(frame, "NO HAND DETECTED", (w // 2 - 120, 40),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 80, 255), 2)

                # ── Hand frame quality bar (top-right) ────────────────────────
                if len(hand_frame_flags) > 0:
                    quality = sum(hand_frame_flags) / len(hand_frame_flags)
                    q_color = (0, 200, 80) if quality >= 0.5 else (0, 100, 255)
                    cv2.putText(frame, f"Hand: {quality:.0%}", (w - 130, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 2)

                # ── Debounce / emit ────────────────────────────────────────────
                if current_sign and current_sign != last_spoken and (now - last_spoken_time) > TTS_COOLDOWN:
                    self.sign_detected_signal.emit(current_sign)
                    last_spoken      = current_sign
                    last_spoken_time = now
                elif not current_sign:
                    last_spoken = ""

                # Qt frame emit
                display_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)

        cap.release()

    def stop(self):
        self._run_flag = False
        # Non-blocking stop system


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

class CommunicatorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #1a1a1a; color: white; font-family: Segoe UI, sans-serif;")
        
        # Core Layout
        main_layout = QHBoxLayout(self)
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
        
        # Avatar GIF/MP4 Queue Layout (up to 3 recent signs)
        self.avatar_container = QWidget()
        self.avatar_layout = QHBoxLayout(self.avatar_container)
        self.avatar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.avatars = []
        for _ in range(3):
            av = AvatarView()
            self.avatar_layout.addWidget(av)
            self.avatars.append(av)
            
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
        try:
            QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ))
        except: pass

    def pause(self):
        """Stops background threads to save resources — Non-blocking."""
        print("[Communicator] Pausing background threads (Non-blocking)...")
        if hasattr(self, 'camera_thread'):
            self.camera_thread._run_flag = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread._run_flag = False
        self.status_label.setText("Threads Paused (Resource Saver)")

    def resume(self):
        """Restarts background threads."""
        print("[Communicator] Resuming background threads...")
        self.init_ai()
        self.status_label.setText("🟢 LSTM Network & Microphone Active")

    def on_sign_detected(self, sign_text):
        if not self.isVisible(): return
        self.add_bubble(sign_text.capitalize(), sender="signer")
        if hasattr(self, 'audio_thread') and self.audio_thread:
            self.audio_thread.speak(sign_text)

    def on_speech_heard(self, text):
        # 1. Show bubble in chat
        self.add_bubble(text, sender="hearing")
        
        # 2. Convert to ASL Gloss for matching
        gloss_tokens = english_to_asl(text)
        self.status_label.setText(f"💬 ASL Gloss: {' '.join(gloss_tokens).upper()}")

        # 3. Match and display signs
        for token in gloss_tokens:
            token = token.lower()
            found_path = None
            
            # Check local dictionary (GIF or MP4)
            for ext in ['.gif', '.mp4']:
                p = os.path.join("sign_gifs", f"{token}{ext}")
                if os.path.exists(p):
                    found_path = p
                    break
            
            # Fallback to dataset folder (picks one sample)
            if not found_path:
                d_path = os.path.join("dataset", token)
                if os.path.exists(d_path):
                    samples = list(Path(d_path).glob("*.mp4")) or list(Path(d_path).glob("*.gif"))
                    # If neither, maybe check .npy preview logic? 
                    # For now just use first video match
                    if samples:
                        found_path = str(samples[0])
            
            if found_path:
                self.play_avatar_sign(found_path)

    def play_avatar_sign(self, path):
        # Shift old signs to the right
        # We need to stop the oldest, move the rest
        paths = [av.current_path for av in self.avatars]
        new_paths = [path, paths[0], paths[1]]

        for i, p in enumerate(new_paths):
            if p:
                self.avatars[i].play(p)
            else:
                self.avatars[i].stop()

    def hide_all_avatar_gifs(self):
        for av in self.avatars:
            av.stop()

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread'): self.camera_thread.stop()
        if hasattr(self, 'audio_thread'): self.audio_thread.stop()
        event.accept()

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Way Sign Language Communicator (LSTM GPU)")
        self.resize(1200, 750)
        self.widget = CommunicatorWidget()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        self.widget.closeEvent(event)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
