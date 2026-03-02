"""
6_video_to_dataset.py
=====================
Converts a folder of sign language videos into the .npy dataset format
used by 2_train_model.py.

EXPECTED INPUT FOLDER STRUCTURE (two options):

  Option A — Labeled subfolders (e.g. WLASL, MS-ASL):
    input_videos/
      hello/
        clip1.mp4
        clip2.mp4
      thank_you/
        clip1.mp4

  Option B — Flat folder with filename prefix as label:
    input_videos/
      hello_001.mp4
      hello_002.mp4
      thankyou_001.mp4

OUTPUT (matching 1_collect_data.py format):
    dataset/
      hello/
        0.npy   ← shape (30, 126)
        1.npy
      thank you/
        0.npy
        ...

Each .npy = 30 frames × 126 features (63 left-hand + 63 right-hand),
with wrist-relative normalization identical to 1_collect_data.py.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30      # frames per sequence (must match training script)
DATASET_PATH    = "dataset"
MIN_DETECTION_CONFIDENCE = 0.5

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic


# =============================================================================
# Keypoint extraction  — IDENTICAL to 1_collect_data.py
# =============================================================================
def extract_keypoints(results):
    """126 wrist-relative features: 63 left-hand + 63 right-hand."""
    if results.left_hand_landmarks:
        lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark])
        lh = (lh - lh[0]).flatten()          # wrist-relative
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark])
        rh = (rh - rh[0]).flatten()          # wrist-relative
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([lh, rh])          # shape: (126,)


# =============================================================================
# Video → sequence of keypoint frames
# =============================================================================
def video_to_sequence(video_path: str, holistic, target_frames: int = SEQUENCE_LENGTH):
    """
    Reads a video, runs MediaPipe on every frame, and returns a
    uniformly-sampled array of shape (target_frames, 126).

    Sampling strategy:
    - If video has >= target_frames frames: sample target_frames evenly
    - If video has fewer frames: pad with zero frames at the end
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return None

    all_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        all_keypoints.append(extract_keypoints(results))
    cap.release()

    if len(all_keypoints) == 0:
        print(f"  [!] No frames extracted from: {video_path}")
        return None

    total = len(all_keypoints)
    arr   = np.array(all_keypoints)   # shape: (total_frames, 126)

    if total >= target_frames:
        # Uniform sampling — pick target_frames evenly spaced indices
        indices = np.linspace(0, total - 1, target_frames, dtype=int)
        return arr[indices]            # shape: (target_frames, 126)
    else:
        # Pad end with zeros
        pad = np.zeros((target_frames - total, 126))
        return np.vstack([arr, pad])   # shape: (target_frames, 126)


# =============================================================================
# Discover videos from a directory
# =============================================================================
def discover_videos(input_dir: str):
    """
    Returns a list of (label, video_path) tuples.
    Supports both labeled-subfolder and flat-with-prefix layouts.
    """
    input_dir = Path(input_dir)
    VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    pairs = []

    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if subdirs:
        # Option A: labeled subfolders
        print(f"[*] Detected labeled-subfolder structure ({len(subdirs)} labels)")
        for subdir in sorted(subdirs):
            label = subdir.name.replace('_', ' ')
            videos = [f for f in subdir.iterdir() if f.suffix.lower() in VIDEO_EXT]
            for v in sorted(videos):
                pairs.append((label, str(v)))
    else:
        # Option B: flat folder — derive label from filename prefix (before last underscore + digits)
        print("[*] Detected flat folder structure")
        videos = [f for f in input_dir.iterdir() if f.suffix.lower() in VIDEO_EXT]
        for v in sorted(videos):
            stem  = v.stem                        # e.g. "hello_001"
            parts = stem.rsplit('_', 1)
            label = parts[0].replace('_', ' ') if len(parts) == 2 and parts[1].isdigit() else stem
            pairs.append((label, str(v)))

    return pairs


# =============================================================================
# Main conversion function
# =============================================================================
def convert(input_dir: str, start_seq_offset: bool = True):
    """
    Converts all videos in input_dir to .npy sequences in dataset/.

    start_seq_offset: if True, new sequences are appended after existing ones
                      (safe to run multiple times without overwriting).
                      Set to False to overwrite from sequence 0.
    """
    pairs = discover_videos(input_dir)
    if not pairs:
        print("[!] No videos found. Check the input folder.")
        return

    labels = sorted(set(label for label, _ in pairs))
    print(f"\n[*] Found {len(pairs)} videos across {len(labels)} labels: {labels}\n")

    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.5,
        static_image_mode=False
    ) as holistic:

        stats = {}

        for label, vpath in pairs:
            # Create output folder
            out_dir = Path(DATASET_PATH) / label
            out_dir.mkdir(parents=True, exist_ok=True)

            # Determine next sequence index
            if start_seq_offset:
                existing = [int(f.stem) for f in out_dir.glob("*.npy") if f.stem.isdigit()]
                seq_idx  = max(existing) + 1 if existing else 0
            else:
                seq_idx = len(list(out_dir.glob("*.npy")))

            rel_path = Path(vpath).name
            print(f"  Processing [{label}]  {rel_path}  →  sequence {seq_idx}")

            sequence = video_to_sequence(vpath, holistic)
            if sequence is None:
                print(f"    [SKIP] Failed to extract frames.")
                continue

            out_path = out_dir / f"{seq_idx}.npy"
            np.save(str(out_path), sequence)
            stats[label] = stats.get(label, 0) + 1
            print(f"    [OK] Saved {sequence.shape}  →  {out_path}")

    print("\n" + "=" * 52)
    print(" CONVERSION COMPLETE")
    print("=" * 52)
    for label, count in sorted(stats.items()):
        existing = len(list((Path(DATASET_PATH) / label).glob("*.npy")))
        print(f"  {label:<20} {count:>3} new   ({existing} total)")
    print(f"\n[*] Dataset folder: {Path(DATASET_PATH).resolve()}")
    print("[*] Ready to run:   python 2_train_model.py")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 52)
    print(" VIDEO → DATASET CONVERTER")
    print(" (wrist-relative normalization, 30-frame sequences)")
    print("=" * 52)

    if len(sys.argv) >= 2:
        input_folder = sys.argv[1]
    else:
        print("\nUsage:")
        print("  python 6_video_to_dataset.py <input_folder>")
        print("\nExample:")
        print("  python 6_video_to_dataset.py C:/Downloads/WLASL/videos")
        print("\nOr enter folder path now:")
        input_folder = input("Input folder: ").strip().strip('"')

    if not os.path.isdir(input_folder):
        print(f"[!] Folder not found: {input_folder}")
        sys.exit(1)

    convert(input_folder)
