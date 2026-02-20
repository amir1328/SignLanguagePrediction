"""
Phase 4: Live Recognition
=========================
Combines OpenCV (DirectShow), MediaPipe, TTS (Pyttsx3) and STT (SpeechRecognition)
for real-time, interactive sign language translation.
"""

import cv2
import pickle
import os
import time
import threading
import queue
import mediapipe as mp

import gtts
import pygame
import speech_recognition as sr


# Configuration
MODEL_FILE         = "sign_model.pkl"
CAMERA_INDEX       = 0
PREDICTION_HISTORY = 7
TTS_COOLDOWN       = 1.5

# UI Colors (BGR)
COLOR_TEXT     = (255, 255, 255)
COLOR_ACCENT   = (0, 215, 255)    # Gold/Yellow
COLOR_PREDICT  = (0, 255, 120)    # Green
COLOR_HEARD    = (255, 150, 0)    # Orange
COLOR_BG       = (30, 30, 30)

# =============================================================================
# 1. Background Workers (Audio I/O)
# =============================================================================

# TTS Queue
tts_q = queue.Queue()

def tts_worker():
    """Reads from queue and plays pre-generated MP3 audio."""
    while True:
        text = tts_q.get()
        if text is None: break
        
        audio_file = os.path.join("audio_signs", f"{text.lower()}.mp3")
        
        if not os.path.exists(audio_file):
            print(f"[TTS Error] Audio file missing for: {text}")
            continue
            
        print(f"[TTS] Playing pre-generated audio for: {text}")
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"[TTS Error] {e}")

def speak(text: str):
    """Flushes queue and queues new text."""
    while not tts_q.empty():
        try: tts_q.get_nowait()
        except queue.Empty: pass
    tts_q.put(text)

# STT Threading
heard_text = ""
heard_lock = threading.Lock()

def stt_worker():
    """Listens in background and updates global string."""
    global heard_text
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        
    print("[*] Microphone listening in background...")
    while True:
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                with heard_lock:
                    heard_text = text
            except Exception:
                pass # Ignore timeouts and unrecognized speech

# =============================================================================
# 2. UI Helpers
# =============================================================================

def draw_hud(frame, sign_text, stt_text):
    """Draws upper prediction bar and lower transcription bar."""
    h, w = frame.shape[:2]
    
    # TOP Bar (Prediction)
    cv2.rectangle(frame, (0, 0), (w, 65), COLOR_BG, -1)
    cv2.line(frame, (0, 65), (w, 65), COLOR_PREDICT, 2)
    
    if sign_text:
        cv2.putText(frame, "Sign:", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
        cv2.putText(frame, sign_text, (105, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_PREDICT, 2)
    else:
        cv2.putText(frame, "Waiting for hand...", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

    # BOTTOM Bar (Heard Speech)
    cv2.rectangle(frame, (0, h-65), (w, h), COLOR_BG, -1)
    cv2.line(frame, (0, h-65), (w, h-65), COLOR_HEARD, 2)
    
    if stt_text:
        display_text = stt_text
    else:
        display_text = "... listening ..."
        
    cv2.putText(frame, "Heard:", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
    cv2.putText(frame, f"\"{display_text}\"", (115, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_HEARD, 2)

# =============================================================================
# 3. Main Application
# =============================================================================

def get_camera():
    """Tries multiple camera backends and indices to find a working stream."""
    backends = [
        ("DirectShow (DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (MSMF)", cv2.CAP_MSMF),
        ("Default (ANY)", cv2.CAP_ANY)
    ]
    indices = [1, 0, 2]  # Try 1 first (usually real webcam if 0 is a virtual camera)

    for index in indices:
        for name, backend in backends:
            print(f"[*] Trying Camera Index {index} with {name}...")
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[✔] SUCCESS: Camera {index} opened with {name}!")
                    return cap
                else:
                    print(f"    [!] Opened, but could not read a frame.")
                    cap.release()
            else:
                print(f"    [!] Failed to open.")
    
    return None

def main():
    print(f"\n{'='*50}")
    print(f" 🖐 LIVE SIGN RECOGNITION (AUDIO INTEGRATED) ")
    print(f"{'='*50}\n")

    # Load Model
    if not os.path.exists(MODEL_FILE):
        print(f"[!] FATAL: Trained model '{MODEL_FILE}' not found.")
        print("    Please run '2_train_model.py' first.")
        return

    print("[*] Loading trained RandomForest model...")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    # Start audio threads
    print("[*] Starting Audio engine (TTS)...")
    threading.Thread(target=tts_worker, daemon=True).start()
    print("[*] Starting Microphone engine (STT)...")
    threading.Thread(target=stt_worker, daemon=True).start()

    # Load Vision
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    print("\n[*] Initializing camera...")
    cap = get_camera()
    if not cap:
        print("\n[!] FATAL: Could not open ANY webcam after trying all backends.")
        print("    Please check your Windows Privacy settings for Camera access,")
        print("    or ensure no other app (like Zoom or the Streamlit app) is using it.")
        return

    history = []
    last_spoken = ""
    last_spoken_time = 0.0

    print(f"\n[READY] Look at the camera window!")
    print(f"        Press 'ESC' or 'Q' in the window to quit.")

    # Main Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        current_sign = ""

        # Tracking Logic
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )

                # Extract 63 features
                features = []
                for lm in hand_lms.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                # Predict
                pred = model.predict([features])[0]
                history.append(pred)
                if len(history) > PREDICTION_HISTORY:
                    history.pop(0)

                # Smooth results via majority vote
                current_sign = max(set(history), key=history.count)
        else:
            # Decay history when no hand is present
            history.append("")
            if len(history) > PREDICTION_HISTORY:
                history.pop(0)
            if history:
                current_sign = max(set(history), key=history.count)

        # TTS Logic
        now = time.time()
        if (current_sign and current_sign != "" and 
            current_sign != last_spoken and 
            (now - last_spoken_time) > TTS_COOLDOWN):
            
            print(f"[TTS Trigger] Queuing speech: {current_sign}")
            speak(current_sign)
            last_spoken = current_sign
            last_spoken_time = now
        elif current_sign == "":
            # Reset so we can speak the same sign again after hand is removed
            last_spoken = ""

        # Get transcription
        with heard_lock:
            stt_display = heard_text

        # Draw Interface
        draw_hud(frame, current_sign, stt_display)
        cv2.imshow("Sign Language Live Recognition", frame)

        # Keyboard Interrupt
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("\n[!] Exiting...")
            break

    # Cleanup
    tts_q.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print("[✔] Successfully closed application.\n")

if __name__ == "__main__":
    main()
