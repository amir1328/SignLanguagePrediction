import cv2
import csv
import os
import time
import mediapipe as mp
import gtts
import imageio

# Configuration
DATASET_FILE    = "dataset.csv"
FRAMES_PER_SIGN = 500
CAMERA_INDEX    = 0

# UI Colors (BGR)
COLOR_TEXT     = (255, 255, 255)
COLOR_ACCENT   = (0, 215, 255)    # Gold/Yellow
COLOR_RECORD   = (0, 70, 255)     # Red
COLOR_BG       = (30, 30, 30)

def draw_header(frame, title, subtitle=None, bg_color=COLOR_BG):
    """Draws a clean, styled header at the top of the frame."""
    h, w = frame.shape[:2]
    # Header background
    cv2.rectangle(frame, (0, 0), (w, 60 if subtitle else 40), bg_color, -1)
    # Bottom accent line
    cv2.line(frame, (0, 60 if subtitle else 40), (w, 60 if subtitle else 40), COLOR_ACCENT, 2)
    
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 1)
    if subtitle:
        cv2.putText(frame, subtitle, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

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
    print(f" 📷 SIGN LANGUAGE DATA COLLECTION  ")
    print(f"{'='*50}\n")

    # 1. Initialize CSV Header if needed
    # Header format: x0_L, y0_L, z0_L ... x20_L, y20_L, z20_L, x0_R, y0_R, z0_R ... x20_R, y20_R, z20_R, label
    header = []
    for hand in ('L', 'R'):
        for i in range(21):
            for axis in ('x', 'y', 'z'):
                header.append(f"{axis}{i}_{hand}")
    header.append("label")
    
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            csv.writer(f).writerow(header)
        print(f"[*] Created new '{DATASET_FILE}'")
    else:
        print(f"[*] Appending to existing '{DATASET_FILE}'")

    # 2. Get Label from User
    while True:
        label = input("\n👉 Enter the sign you want to record (or 'q' to quit): ").strip()
        if not label: continue
        if label.lower() == 'q':
            print("Exiting...")
            return
            
        # 3. Generate Audio for the Label
        audio_dir = "audio_signs"
        gif_dir   = "sign_gifs"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            
        audio_path = os.path.join(audio_dir, f"{label.lower()}.mp3")
        gif_path   = os.path.join(gif_dir, f"{label.lower()}.gif")
        if not os.path.exists(audio_path):
            print(f"[*] Generating speech audio for '{label}'...")
            try:
                tts = gtts.gTTS(text=label, lang='en')
                tts.save(audio_path)
                print(f"    [✔] Saved audio to {audio_path}")
            except Exception as e:
                print(f"    [!] Warning: Could not generate audio: {e}")
        else:
            print(f"[*] Audio for '{label}' already exists.")
        
        break

    # 4. Initialize Camera & MediaPipe
    print("\n[*] Initializing camera...")
    cap = get_camera()
    if not cap:
        print("\n[!] FATAL: Could not open ANY webcam after trying all backends.")
        print("    Please check your Windows Privacy settings for Camera access,")
        print("    or ensure no other app (like Zoom or the Streamlit app) is using it.")
        return

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    collected_frames = 0
    is_recording     = False
    gif_frames       = []
    
    print(f"\n[READY] Camera active.")
    print(f"        Press 'R' to Start Recording")
    print(f"        Press 'ESC' to Cancel")

    # 4. Collection Loop
    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        while cap.isOpened() and collected_frames < FRAMES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Process hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # We need exactly 126 features (63 per hand * 2 hands)
            features = []
            # MediaPipe doesn't strictly guarantee index 0 is left and index 1 is right, 
            # but we process up to 2 hands found in the frame.
            if result.multi_hand_landmarks:
                for i in range(min(2, len(result.multi_hand_landmarks))):
                    hand_lms = result.multi_hand_landmarks[i]
                    mp_draw.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )
                    # Extract 63 features for this hand
                    for lm in hand_lms.landmark:
                        features.extend([lm.x, lm.y, lm.z])
            
            # Pad to 126 features if only 1 hand or no hands are detected
            while len(features) < 126:
                features.append(0.0)

            # Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n[!] Cancelled by user.")
                break
            elif key == ord('r') or key == ord('R'):
                if not is_recording:
                    is_recording = True
                    collected_frames = 0
                    gif_frames = []
                    print("\n[REC] Recording started...")

            # Save data if recording
            if is_recording: # We always have 126 features due to padding
                writer.writerow(features + [label])
                
                # Capture every 5th frame for the GIF, resize to 320x240 to save space
                if collected_frames % 5 == 0:
                    small_frame = cv2.resize(rgb_frame, (320, 240))
                    gif_frames.append(small_frame)
                    
                collected_frames += 1

            # UI Overlays
            if not is_recording:
                draw_header(frame, f"Sign: {label}", "Press 'R' to start recording | 'ESC' to quit")
            else:
                pct = collected_frames / FRAMES_PER_SIGN
                draw_header(frame, f"Recording: {label}", f"Progress: {collected_frames}/{FRAMES_PER_SIGN} frames", bg_color=COLOR_RECORD)
                
                # Progress bar at bottom
                bar_width = int(w * pct)
                cv2.rectangle(frame, (0, h-10), (bar_width, h), COLOR_ACCENT, -1)

            cv2.imshow("Sign Language Data Collection", frame)

    # 5. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if collected_frames >= FRAMES_PER_SIGN:
        print(f"\n[✔] Successfully recorded {FRAMES_PER_SIGN} frames for '{label}'.")
        print(f"    Data saved to {DATASET_FILE}")
        
        if gif_frames:
            print(f"[*] Generating Avatar GIF ({len(gif_frames)} frames)...")
            try:
                imageio.mimsave(gif_path, gif_frames, fps=15)
                print(f"    [✔] Saved GIF to {gif_path}")
            except Exception as e:
                print(f"    [!] Error saving GIF: {e}")

if __name__ == "__main__":
    main()
