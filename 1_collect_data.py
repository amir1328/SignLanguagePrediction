import cv2
import numpy as np
import os
import time
import mediapipe as mp
import gtts
import imageio

# Configuration
DATASET_PATH    = os.path.join("dataset")
NO_SEQUENCES    = 15 # Number of videos per sign
SEQUENCE_LENGTH = 30 # Frames per video
CAMERA_INDEX    = 1

# No face indices needed anymore

# UI Colors (BGR)
COLOR_TEXT     = (255, 255, 255)
COLOR_ACCENT   = (0, 215, 255)    # Gold/Yellow
COLOR_RECORD   = (0, 70, 255)     # Red
COLOR_BG       = (30, 30, 30)

def draw_header(frame, title, subtitle=None, bg_color=COLOR_BG):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60 if subtitle else 40), bg_color, -1)
    cv2.line(frame, (0, 60 if subtitle else 40), (w, 60 if subtitle else 40), COLOR_ACCENT, 2)
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 1)
    if subtitle:
        cv2.putText(frame, subtitle, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

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

def extract_keypoints(results):
    """Extracts exactly 126 features: 63 (Left) + 63 (Right)"""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([lh, rh])

def main():
    print(f"\n{'='*50}")
    print(f" 📷 SEQUENTIAL LSTM DATA COLLECTION  ")
    print(f"{'='*50}\n")
    
    label = input("👉 Enter the temporal sign you want to record (or 'q' to quit): ").strip().lower()
    if not label or label == 'q':
        return

    # Folder setup
    sign_path = os.path.join(DATASET_PATH, label)
    audio_dir = "audio_signs"
    gif_dir   = "sign_gifs"
    
    for path in [sign_path, audio_dir, gif_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Audio/GIF Gen
    audio_path = os.path.join(audio_dir, f"{label}.mp3")
    gif_path   = os.path.join(gif_dir, f"{label}.gif")
    
    if not os.path.exists(audio_path):
        print(f"[*] Generating speech audio for '{label}'...")
        try:
            tts = gtts.gTTS(text=label, lang='en')
            tts.save(audio_path)
            print(f"    [✔] Saved audio to {audio_path}")
        except Exception as e:
            print(f"    [!] Warning: Could not generate audio: {e}")

    # Initialization
    print("\n[*] Initializing camera...")
    cap = get_camera()
    if not cap:
        print("\n[!] FATAL: Could not open webcam.")
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    gif_frames = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print(f"\n[READY] Camera active.")
        print(f"        Get ready to record {NO_SEQUENCES} sequences.")
        print(f"        Press 'R' to Start Sequence Collection")
        
        # Wait for 'r' key
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            draw_header(frame, f"Sign: {label.capitalize()}", "Press 'R' to start recording sequences | 'ESC' to quit")
            cv2.imshow("LSTM Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        # Start Collecting Sequences
        for sequence in range(NO_SEQUENCES):
            # Create a 2 second waiting period between sequence actions so the user can reset hands
            for w in range(30):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                draw_header(frame, f"GET READY FOR SEQUENCE #{sequence+1}", f"Starting in {2 - (w//15)}...", bg_color=(0, 100, 200))
                cv2.imshow("LSTM Collection", frame)
                cv2.waitKey(30)
                
            sequence_data = []
            
            # Record 30 continuous frames
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Draw landmarks for visual feedback
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                # We won't draw face mesh as it gets too cluttered, just verify we extract it
                
                # Capture GIF frames during the VERY FIRST sequence only (resize to save space)
                if sequence == 0 and frame_num % 2 == 0:
                    small_frame = cv2.resize(rgb_frame, (320, 240))
                    gif_frames.append(small_frame)

                # Extract and store keypoints
                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)

                # UI Overlay
                draw_header(frame, f"Recording: {label.capitalize()}", f"Sequence {sequence}/{NO_SEQUENCES} | Frame {frame_num}/{SEQUENCE_LENGTH}", bg_color=COLOR_RECORD)
                
                # Progress bar at bottom
                pct = frame_num / SEQUENCE_LENGTH
                cv2.rectangle(frame, (0, h-10), (int(w * pct), h), COLOR_ACCENT, -1)

                cv2.imshow("LSTM Collection", frame)
                if cv2.waitKey(10) & 0xFF == 27:
                    print("Cancelled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Save the numpy sequence (30 frames of 225 coordinates each)
            npy_path = os.path.join(sign_path, str(sequence))
            np.save(npy_path, np.array(sequence_data))
            print(f"    [+] Saved Sequence {sequence}")

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[✔] Successfully recorded {NO_SEQUENCES} sequences for '{label}'.")
    
    if gif_frames:
        print(f"[*] Generating Avatar GIF ({len(gif_frames)} frames)...")
        try:
            imageio.mimsave(gif_path, gif_frames, fps=15)
            print(f"    [✔] Saved GIF to {gif_path}")
        except Exception as e:
            print(f"    [!] Error saving GIF: {e}")

if __name__ == "__main__":
    main()
