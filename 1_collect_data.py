import cv2
import numpy as np
import os
import json
import time
import mediapipe as mp
import gtts
import imageio

# Configuration
DATASET_PATH    = os.path.join("dataset")
NO_SEQUENCES    = 50   # Number of videos per sign
SEQUENCE_LENGTH = 30   # Frames per video
CAMERA_INDEX    = 1
MIN_HAND_FRAMES = 20   # Min frames with detected hand per sequence (out of 30)
MAX_RETRIES     = 3    # Max auto-retakes for rejected sequences
ZONE_MARGIN     = 0.15 # ± margin around median wrist position for zone boundary
SIGN_ZONES_FILE = "sign_zones.json"

# UI Colors (BGR)
COLOR_TEXT    = (255, 255, 255)
COLOR_ACCENT  = (0, 215, 255)    # Gold/Yellow
COLOR_RECORD  = (0, 70, 255)     # Red
COLOR_BG      = (30, 30, 30)
COLOR_OK      = (0, 200, 80)     # Green
COLOR_WARN    = (0, 100, 255)    # Orange-red

def draw_header(frame, title, subtitle=None, bg_color=COLOR_BG):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60 if subtitle else 40), bg_color, -1)
    cv2.line(frame, (0, 60 if subtitle else 40), (w, 60 if subtitle else 40), COLOR_ACCENT, 2)
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 1)
    if subtitle:
        cv2.putText(frame, subtitle, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

def draw_hand_status(frame, has_hand, quality_pct=None):
    """Draw live hand detection status indicator in top-right corner."""
    h, w = frame.shape[:2]
    status_text = "Hand: OK" if has_hand else "Hand: --"
    color = COLOR_OK if has_hand else COLOR_WARN
    cv2.putText(frame, status_text, (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if quality_pct is not None:
        q_color = COLOR_OK if quality_pct >= 70 else COLOR_WARN
        cv2.putText(frame, f"Quality: {quality_pct:.0f}%", (w - 145, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 1)

def draw_stability_warning(frame, is_stable):
    """Show wrist stability indicator."""
    h, w = frame.shape[:2]
    if not is_stable:
        cv2.putText(frame, "! HAND MOVING FAST", (w - 200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARN, 2)

def get_camera():
    backends = [
        ("DirectShow (DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (MSMF)", cv2.CAP_MSMF),
        ("Default (ANY)", cv2.CAP_ANY)
    ]
    for index in [1, 0, 2]:
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
                cap.release()
    return None

def extract_keypoints(results):
    """Extracts 126 wrist-relative features: 63 (Left) + 63 (Right)."""
    if results.left_hand_landmarks:
        lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark])
        lh = (lh - lh[0]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark])
        rh = (rh - rh[0]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([lh, rh])

def get_wrist_pos(results):
    """Return the absolute (x, y) wrist position (normalized 0-1) of whichever hand is visible."""
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark[0]  # Wrist = index 0
        return (lm.x, lm.y)
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark[0]
        return (lm.x, lm.y)
    return None

def has_any_hand(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def wrist_stable(prev_pos, curr_pos, threshold=0.12):
    """Check that the wrist didn't jump more than `threshold` in one frame."""
    if prev_pos is None or curr_pos is None:
        return True
    dx = abs(curr_pos[0] - prev_pos[0])
    dy = abs(curr_pos[1] - prev_pos[1])
    return (dx + dy) < threshold

def compute_zone(wrist_positions):
    """Compute ± ZONE_MARGIN bounding zone from a list of (x, y) positions."""
    xs = [p[0] for p in wrist_positions]
    ys = [p[1] for p in wrist_positions]
    mx, my = float(np.median(xs)), float(np.median(ys))
    return {
        "x": [round(max(0.0, mx - ZONE_MARGIN), 3), round(min(1.0, mx + ZONE_MARGIN), 3)],
        "y": [round(max(0.0, my - ZONE_MARGIN), 3), round(min(1.0, my + ZONE_MARGIN), 3)]
    }

def load_sign_zones():
    if os.path.exists(SIGN_ZONES_FILE):
        with open(SIGN_ZONES_FILE) as f:
            return json.load(f)
    return {}

def save_sign_zones(zones):
    with open(SIGN_ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"    [✔] Updated {SIGN_ZONES_FILE}")

def record_sequence(cap, holistic, mp_holistic, mp_drawing, label, seq_idx, total_seqs):
    """
    Record one 30-frame sequence. Returns (sequence_data, wrist_positions, quality_pct)
    or (None, None, quality_pct) if the sequence is rejected.
    """
    # 2-second countdown
    for w in range(30):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        countdown = 2 - (w // 15)
        draw_header(frame, f"GET READY #{seq_idx+1}/{total_seqs}",
                    f"Starting in {countdown}s... Lower hands, then sign!", bg_color=(0, 100, 200))
        cv2.imshow("LSTM Collection", frame)
        cv2.waitKey(30)

    sequence_data  = []
    wrist_positions = []
    hand_frames     = 0
    prev_wrist      = None
    teleport_frames = 0

    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        hand_visible = has_any_hand(results)
        wrist_pos    = get_wrist_pos(results)
        is_stable    = wrist_stable(prev_wrist, wrist_pos)

        if hand_visible:
            hand_frames += 1
        if not is_stable:
            teleport_frames += 1
        if wrist_pos:
            wrist_positions.append(wrist_pos)
        prev_wrist = wrist_pos

        # Running quality metric
        quality_so_far = (hand_frames / max(frame_num + 1, 1)) * 100

        # UI
        draw_header(frame, f"Recording: {label.capitalize()}",
                    f"Seq {seq_idx+1}/{total_seqs} | Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                    bg_color=COLOR_RECORD)
        draw_hand_status(frame, hand_visible, quality_so_far)
        draw_stability_warning(frame, is_stable)

        # Progress bar
        pct = frame_num / SEQUENCE_LENGTH
        cv2.rectangle(frame, (0, h - 10), (int(w * pct), h), COLOR_ACCENT, -1)

        keypoints = extract_keypoints(results)
        sequence_data.append(keypoints)

        cv2.imshow("LSTM Collection", frame)
        if cv2.waitKey(10) & 0xFF == 27:
            return None, None, 0

    final_quality = (hand_frames / SEQUENCE_LENGTH) * 100

    # Validate: reject if too few hand frames OR too many teleports
    if hand_frames < MIN_HAND_FRAMES:
        print(f"    [✗] Sequence REJECTED — only {hand_frames}/{SEQUENCE_LENGTH} hand frames ({final_quality:.0f}%). Retrying...")
        return None, wrist_positions, final_quality
    if teleport_frames > 5:
        print(f"    [✗] Sequence REJECTED — wrist unstable ({teleport_frames} large jumps). Retrying...")
        return None, wrist_positions, final_quality

    print(f"    [✓] Sequence accepted — quality {final_quality:.0f}% ({hand_frames}/{SEQUENCE_LENGTH} frames)")
    return sequence_data, wrist_positions, final_quality

def main():
    print(f"\n{'='*50}")
    print(f" 📷 SEQUENTIAL LSTM DATA COLLECTION  ")
    print(f"{'='*50}\n")

    label = input("👉 Enter the sign you want to record (or 'q' to quit): ").strip().lower()
    if not label or label == 'q':
        return

    sign_path = os.path.join(DATASET_PATH, label)
    audio_dir = "audio_signs"
    gif_dir   = "sign_gifs"

    for path in [sign_path, audio_dir, gif_dir]:
        os.makedirs(path, exist_ok=True)

    # Determine next sequence index (don't overwrite existing data)
    existing = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
    start_idx = len(existing)
    print(f"[*] Existing sequences: {start_idx}. Recording {NO_SEQUENCES} more → total will be {start_idx + NO_SEQUENCES}.")

    # Audio gen
    audio_path = os.path.join(audio_dir, f"{label}.mp3")
    if not os.path.exists(audio_path):
        print(f"[*] Generating audio for '{label}'...")
        try:
            gtts.gTTS(text=label, lang='en').save(audio_path)
            print(f"    [✔] Saved audio → {audio_path}")
        except Exception as e:
            print(f"    [!] Warning: Could not generate audio: {e}")

    print("\n[*] Initializing camera...")
    cap = get_camera()
    if not cap:
        print("\n[!] FATAL: Could not open webcam.")
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils

    gif_frames      = []
    all_wrist_pos   = []   # Aggregate across all accepted sequences for zone computation
    saved_sequences = 0

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        print(f"\n[READY] Camera active. Press 'R' to start | ESC to quit.")
        # Wait for 'R'
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            draw_header(frame, f"Sign: {label.capitalize()}", "Press 'R' to start | ESC to quit")
            # Show live hand preview even here
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            draw_hand_status(frame, has_any_hand(res))
            cv2.imshow("LSTM Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('r'), ord('R')):
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        sequence_idx = start_idx
        target_idx   = start_idx

        while saved_sequences < NO_SEQUENCES:
            seq_data, wrist_pos, quality = record_sequence(
                cap, holistic, mp_holistic, mp_drawing,
                label, saved_sequences, NO_SEQUENCES
            )

            if seq_data is None and quality == 0:
                # ESC pressed
                break

            if seq_data is None:
                # Rejected — show retake UI briefly
                for _ in range(20):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    draw_header(frame, "RETAKING...", "Prepare again!", bg_color=(0, 50, 150))
                    cv2.imshow("LSTM Collection", frame)
                    cv2.waitKey(30)
                continue

            # Accepted sequence
            if wrist_pos:
                all_wrist_pos.extend(wrist_pos)

            # Save GIF frames from very first accepted sequence
            if saved_sequences == 0:
                # We'll grab frames from the saved data display frame
                pass  # GIF will be a simple note — skip here for speed

            npy_path = os.path.join(sign_path, str(target_idx))
            np.save(npy_path, np.array(seq_data))
            saved_sequences += 1
            target_idx      += 1
            print(f"    [+] Saved sequence {target_idx} ({saved_sequences}/{NO_SEQUENCES})")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[✔] Recorded {saved_sequences}/{NO_SEQUENCES} sequences for '{label}'.")

    # Compute and save zone for this sign
    if all_wrist_pos:
        zone = compute_zone(all_wrist_pos)
        zones = load_sign_zones()
        zones[label] = zone
        save_sign_zones(zones)
        print(f"    [📍] Zone for '{label}': x={zone['x']}, y={zone['y']}")
    else:
        print("    [!] No wrist positions recorded — zone not updated.")

if __name__ == "__main__":
    main()
