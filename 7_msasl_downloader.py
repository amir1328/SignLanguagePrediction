"""
7_msasl_downloader.py
=====================
Downloads clips from the MS-ASL dataset JSON files, runs them through
MediaPipe, and saves .npy sequences in the dataset/ folder format.

Requirements:
  pip install yt-dlp moviepy mediapipe

Usage:
  python 7_msasl_downloader.py                      <- prompts for options
  python 7_msasl_downloader.py --words hello,help   <- specific words
  python 7_msasl_downloader.py --top 20             <- top 20 most common words
  python 7_msasl_downloader.py --label 0            <- MS-ASL label number
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
MSASL_DIR      = r"C:\Users\AMEER AKBAR\Downloads\MS-ASL\MS-ASL"
DATASET_PATH   = "dataset"
SEQUENCE_LENGTH = 30
CLIPS_PER_SIGN  = 50     # max sequences to collect per sign (matches 1_collect_data.py)
MIN_DETECTION_CONFIDENCE = 0.5

mp_holistic = mp.solutions.holistic


# =============================================================================
# Keypoint extraction  — IDENTICAL to 1_collect_data.py
# =============================================================================
def extract_keypoints(results):
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


# =============================================================================
# Video → 30-frame keypoint sequence
# =============================================================================
def video_to_sequence(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(extract_keypoints(holistic.process(rgb)))
    cap.release()
    if not frames:
        return None
    arr = np.array(frames)
    total = len(arr)
    if total >= SEQUENCE_LENGTH:
        indices = np.linspace(0, total - 1, SEQUENCE_LENGTH, dtype=int)
        return arr[indices]
    else:
        pad = np.zeros((SEQUENCE_LENGTH - total, 126))
        return np.vstack([arr, pad])


# =============================================================================
# Download a YouTube clip using yt-dlp + moviepy trim
# =============================================================================
def download_clip(url, start_time, end_time, out_path):
    """Download and trim a YouTube clip. Returns True on success."""
    try:
        import yt_dlp
        from moviepy.editor import VideoFileClip
    except ImportError as e:
        print(f"  [!] Missing library: {e}")
        print("      Run: pip install yt-dlp moviepy")
        return False

    tmp_dir  = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "raw.%(ext)s")

    ydl_opts = {
        # Use a pre-merged progressive MP4 — no ffmpeg required
        "format": "best[ext=mp4][height<=480]/best[ext=mp4]/best[height<=480]/best",
        "outtmpl": tmp_file,
        "quiet": True,
        "no_warnings": True,
    }

    # Normalise URL
    if not url.startswith("http"):
        url = "https://" + url

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded file
        raw = next(
            (os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
             if f.startswith("raw")), None
        )
        if not raw or not os.path.exists(raw):
            return False

        # Trim with moviepy (add 0.5s padding on each side for safety)
        clip = VideoFileClip(raw).subclip(
            max(0, start_time - 0.3),
            end_time + 0.3
        )
        clip.write_videofile(out_path, verbose=False, logger=None)
        clip.close()
        return True

    except Exception as e:
        print(f"  [!] Download failed: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Load MS-ASL JSON entries
# =============================================================================
def load_entries(json_files=None, words=None, labels=None, top_n=None):
    if json_files is None:
        json_files = [
            os.path.join(MSASL_DIR, "MSASL_train.json"),
            os.path.join(MSASL_DIR, "MSASL_val.json"),
        ]

    all_entries = []
    for jf in json_files:
        if os.path.exists(jf):
            with open(jf, "r", encoding="utf-8") as f:
                all_entries.extend(json.load(f))

    if not all_entries:
        print("[!] No entries loaded. Check MSASL_DIR path.")
        return []

    # Filter
    if labels is not None:
        all_entries = [e for e in all_entries if e["label"] in labels]
    if words is not None:
        words_lower = [w.lower() for w in words]
        all_entries = [e for e in all_entries if e["text"].lower() in words_lower]

    if top_n is not None and words is None and labels is None:
        counts  = Counter(e["text"].lower() for e in all_entries)
        top_words = [w for w, _ in counts.most_common(top_n)]
        all_entries = [e for e in all_entries if e["text"].lower() in top_words]

    return all_entries


# =============================================================================
# Main download + convert loop
# =============================================================================
def run(words=None, labels=None, top_n=None, max_per_sign=CLIPS_PER_SIGN):
    entries = load_entries(words=words, labels=labels, top_n=top_n)
    if not entries:
        print("[!] No entries matched your filter.")
        return

    # Group by sign text
    by_sign = {}
    for e in entries:
        key = e["text"].lower()
        by_sign.setdefault(key, []).append(e)

    print(f"\n[*] Signs to process: {sorted(by_sign.keys())}")
    print(f"[*] Max clips per sign: {max_per_sign}\n")

    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.5,
    ) as holistic:

        stats = {}
        for sign_text, sign_entries in sorted(by_sign.items()):

            out_dir = Path(DATASET_PATH) / sign_text
            out_dir.mkdir(parents=True, exist_ok=True)

            existing = [int(f.stem) for f in out_dir.glob("*.npy") if f.stem.isdigit()]
            seq_idx  = max(existing) + 1 if existing else 0

            if seq_idx >= max_per_sign:
                print(f"  [{sign_text}] Already has {seq_idx} sequences, skipping.")
                continue

            print(f"\n{'='*52}")
            print(f"  Sign: {sign_text}  (need {max_per_sign - seq_idx} more)")
            print(f"{'='*52}")

            saved = 0
            for entry in sign_entries:
                if seq_idx >= max_per_sign:
                    break

                url        = entry["url"]
                start_time = entry.get("start_time", 0)
                end_time   = entry.get("end_time", start_time + 3)

                print(f"  Clip {seq_idx}: {url}  [{start_time:.1f}s – {end_time:.1f}s]")

                tmp_clip = tempfile.mktemp(suffix=".mp4")
                ok = download_clip(url, start_time, end_time, tmp_clip)
                if not ok:
                    print("    [SKIP] Download failed.")
                    continue

                sequence = video_to_sequence(tmp_clip, holistic)
                os.remove(tmp_clip)

                if sequence is None:
                    print("    [SKIP] No landmarks detected.")
                    continue

                out_path = out_dir / f"{seq_idx}.npy"
                np.save(str(out_path), sequence)
                print(f"    [OK]  Saved {sequence.shape}  →  {out_path}")
                seq_idx += 1
                saved  += 1

            stats[sign_text] = saved

    print("\n" + "=" * 52)
    print(" DOWNLOAD COMPLETE")
    print("=" * 52)
    for sign, count in sorted(stats.items()):
        total = len(list((Path(DATASET_PATH) / sign).glob("*.npy")))
        print(f"  {sign:<20} +{count:>3} new   ({total} total)")
    print(f"\n[*] Dataset: {Path(DATASET_PATH).resolve()}")
    print("[*] Ready:   python 2_train_model.py")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MS-ASL → dataset converter")
    parser.add_argument("--words",  type=str, help="Comma-separated words, e.g. hello,help,water")
    parser.add_argument("--top",    type=int, help="Download top N most common signs")
    parser.add_argument("--label",  type=int, nargs="+", help="MS-ASL numeric label(s)")
    parser.add_argument("--max",    type=int, default=CLIPS_PER_SIGN, help=f"Max clips per sign (default {CLIPS_PER_SIGN})")
    args = parser.parse_args()

    words  = [w.strip() for w in args.words.split(",")] if args.words else None
    labels = args.label

    if words is None and labels is None and args.top is None:
        print("MS-ASL Dataset Downloader")
        print("="*40)
        print("Available options:")
        print("  1) Download specific words (e.g. hello, help, water)")
        print("  2) Download top N most common signs")
        print()
        choice = input("Enter words (comma-separated) or a number for top-N: ").strip()
        if choice.isdigit():
            args.top = int(choice)
        else:
            words = [w.strip() for w in choice.split(",")]

    run(words=words, labels=labels, top_n=args.top, max_per_sign=args.max)
