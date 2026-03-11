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
import threading
import concurrent.futures
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import Counter
from gtts import gTTS

# ── Config ───────────────────────────────────────────────────────────────────
MSASL_DIR                = "MS-ASL"   # Official MS-ASL (1000 signs, ~25k clips)
VALID25_DIR              = "MSASL-valid-dataset-downloader-main"  # Pre-validated 25% (156 signs, better URLs)
DATASET_PATH             = "dataset"
SEQUENCE_LENGTH          = 30
CLIPS_PER_SIGN           = 50
PARALLEL_WORKERS         = 4    # Parallel download+process threads (tune to your internet speed)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_HAND_FRAMES          = 20   # Reject sequences with fewer hand-detected frames
MAX_TELEPORT_FRAMES      = 5    # Reject sequences with too many large wrist jumps
ZONE_MARGIN              = 0.15 # ± zone boundary around median wrist position
SIGN_ZONES_FILE          = "sign_zones.json"

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

def has_any_hand(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def get_wrist_pos(results):
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark[0]
        return (lm.x, lm.y)
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark[0]
        return (lm.x, lm.y)
    return None

def wrist_stable(prev, curr, threshold=0.12):
    if prev is None or curr is None:
        return True
    return abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) < threshold

# ───────────────────────────────────────────────────────────────────
# Sign Zone helpers
# ───────────────────────────────────────────────────────────────────
def load_sign_zones():
    if os.path.exists(SIGN_ZONES_FILE):
        with open(SIGN_ZONES_FILE) as f:
            return json.load(f)
    return {}

def save_sign_zones(zones):
    with open(SIGN_ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"  [✔] Updated {SIGN_ZONES_FILE}")

def compute_zone(wrist_positions):
    xs = [p[0] for p in wrist_positions]
    ys = [p[1] for p in wrist_positions]
    mx, my = float(np.median(xs)), float(np.median(ys))
    return {
        "x": [round(max(0.0, mx - ZONE_MARGIN), 3), round(min(1.0, mx + ZONE_MARGIN), 3)],
        "y": [round(max(0.0, my - ZONE_MARGIN), 3), round(min(1.0, my + ZONE_MARGIN), 3)]
    }



# =============================================================================
# Video → 30-frame keypoint sequence
# =============================================================================
def video_to_sequence(video_path, holistic):
    """Convert a video to a 30-frame keypoint sequence with quality validation.

    Returns (sequence_array, wrist_positions) if quality passes,
    or (None, []) if the clip is too noisy / has too few hand detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []

    raw_keypoints = []
    hand_flags    = []
    wrist_positions = []
    prev_wrist    = None
    teleport_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        kp      = extract_keypoints(results)
        hv      = has_any_hand(results)
        wp      = get_wrist_pos(results)

        raw_keypoints.append(kp)
        hand_flags.append(hv)
        if wp:
            wrist_positions.append(wp)
        if not wrist_stable(prev_wrist, wp):
            teleport_count += 1
        prev_wrist = wp

    cap.release()

    if not raw_keypoints:
        return None, []

    # ── Quality gate 1: minimum hand-detected frames ──────────────────────
    total = len(raw_keypoints)
    needed_hand = max(1, int(total * (MIN_HAND_FRAMES / SEQUENCE_LENGTH)))
    if sum(hand_flags) < needed_hand:
        print(f"    [REJECT] Only {sum(hand_flags)}/{total} hand frames — too few detections.")
        return None, []

    # ── Quality gate 2: wrist stability ───────────────────────────────
    if teleport_count > MAX_TELEPORT_FRAMES:
        print(f"    [REJECT] {teleport_count} wrist teleport frames — unstable clip.")
        return None, []

    # ── Resample to exactly SEQUENCE_LENGTH frames ───────────────────────
    arr = np.array(raw_keypoints)
    if total >= SEQUENCE_LENGTH:
        indices = np.linspace(0, total - 1, SEQUENCE_LENGTH, dtype=int)
        return arr[indices], wrist_positions
    else:
        pad = np.zeros((SEQUENCE_LENGTH - total, 126))
        return np.vstack([arr, pad]), wrist_positions


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
        clip.write_videofile(out_path, audio=False, verbose=False, logger=None)
        clip.close()
        return True

    except Exception as e:
        print(f"  [!] Download failed: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _normalise_entry(e):
    """Ensure every entry has a lowercase 'text' key regardless of source format."""
    if 'text' not in e or not e['text']:
        e = dict(e)  # don't mutate original
        e['text'] = (e.get('clean_text') or e.get('org_text') or '').lower()
    else:
        e = dict(e)
        e['text'] = e['text'].lower()
    return e


def _load_json_source(json_files):
    """Load entries from a list of JSON file paths, normalising text fields."""
    entries = []
    for jf in json_files:
        if os.path.exists(jf):
            with open(jf, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if isinstance(item, dict):
                        entries.append(_normalise_entry(item))
    return entries


def load_entries(json_files=None, words=None, labels=None, top_n=None):
    """
    Smart dual-source loader:
      1. Tries MSASL-valid-25% first (pre-verified URLs, 156 signs).
      2. For any word NOT found there, falls back to official MS-ASL (1000 signs).

    This maximises URL success rate while still covering rare signs.
    """
    words_lower = [w.lower() for w in words] if words else None

    # ── Source 1: valid-25% (pre-validated URLs, clean_text field) ────────────
    valid25_files = [
        os.path.join(VALID25_DIR, "MSASL_TRAIN25.json"),
        os.path.join(VALID25_DIR, "MSASL_VAL25.json"),
        os.path.join(VALID25_DIR, "MASL_TEST25.json"),
    ]
    valid25_entries = _load_json_source(valid25_files)
    valid25_signs   = set(e['text'] for e in valid25_entries if e['text'])

    # ── Source 2: official MS-ASL (full 1000-sign vocabulary) ─────────────────
    official_files = [
        os.path.join(MSASL_DIR, "MSASL_train.json"),
        os.path.join(MSASL_DIR, "MSASL_val.json"),
    ]
    official_entries = _load_json_source(official_files)

    # ── Merge: use valid-25% where possible, official as fallback ─────────────
    all_entries = []
    covered_by_valid25 = set()
    fallback_signs     = set()

    if words_lower:
        for word in words_lower:
            v25 = [e for e in valid25_entries if e['text'] == word]
            if v25:
                all_entries.extend(v25)
                covered_by_valid25.add(word)
            else:
                off = [e for e in official_entries if e['text'] == word]
                if off:
                    all_entries.extend(off)
                    fallback_signs.add(word)
                else:
                    print(f"  [!] '{word}' not found in either dataset.")
    elif labels is not None:
        all_entries = [e for e in (valid25_entries + official_entries) if e.get('label') in labels]
    elif top_n is not None:
        counts    = Counter(e['text'] for e in valid25_entries if e['text'])
        top_words = [w for w, _ in counts.most_common(top_n)]
        all_entries = [e for e in valid25_entries if e['text'] in top_words]
    else:
        # Default: everything from valid-25%, supplement with official for missing signs
        all_entries   = list(valid25_entries)
        covered_words = valid25_signs
        extra = [e for e in official_entries if e['text'] not in covered_words]
        all_entries.extend(extra)

    if not all_entries:
        print("[!] No entries found. Check that MS-ASL/ and MSASL-valid-dataset-downloader-main/ exist.")
        return []

    if covered_by_valid25:
        print(f"  [✅] Found in valid-25% (better URLs): {sorted(covered_by_valid25)}")
    if fallback_signs:
        print(f"  [⚠️] Falling back to official MS-ASL  : {sorted(fallback_signs)}")

    return all_entries


# =============================================================================
# Per-clip worker — each thread runs this with its OWN MediaPipe instance
# =============================================================================
def _process_one_clip(entry, out_dir, seq_idx_holder, lock,
                      wrist_pos_list, gift_first_holder, sign_text, max_per_sign):
    """
    Download one clip, run MediaPipe, save .npy if quality passes.
    Returns True if a sequence was saved, False otherwise.
    Use a lock to protect the shared seq_idx counter.
    """
    # Fast pre-check: already enough sequences?
    with lock:
        if seq_idx_holder[0] >= max_per_sign:
            return False

    url        = entry["url"]
    start_time = float(entry.get("start_time", 0))
    end_time   = float(entry.get("end_time", start_time + 3))

    tmp_clip = tempfile.mktemp(suffix=".mp4")
    ok = download_clip(url, start_time, end_time, tmp_clip)
    if not ok:
        return False

    # Each thread gets its own Holistic (not thread-safe to share)
    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.5,
    ) as holistic:
        sequence, wrist_pos = video_to_sequence(tmp_clip, holistic)

    # Save preview GIF for the sign (first successful clip only)
    if sequence is not None:
        with lock:
            is_first = gift_first_holder[0]
            gift_first_holder[0] = False
        if is_first:
            gif_dir  = Path("sign_gifs")
            gif_dir.mkdir(exist_ok=True)
            gif_path = gif_dir / f"{sign_text}.mp4"
            if not gif_path.exists():
                shutil.copy2(tmp_clip, str(gif_path))
                print(f"  [+] Saved preview video: {gif_path}")

    try:
        os.remove(tmp_clip)
    except Exception:
        pass

    if sequence is None:
        return False

    # Thread-safe index allocation + save
    with lock:
        if seq_idx_holder[0] >= max_per_sign:
            return False
        idx = seq_idx_holder[0]
        seq_idx_holder[0] += 1

    out_path = out_dir / f"{idx}.npy"
    np.save(str(out_path), sequence)
    print(f"    [✓] Saved seq {idx:03d}  shape={sequence.shape}")

    with lock:
        wrist_pos_list.extend(wrist_pos)

    return True


# =============================================================================
# Main download + convert loop
# =============================================================================
def run(words=None, labels=None, top_n=None,
        max_per_sign=CLIPS_PER_SIGN, workers=PARALLEL_WORKERS):
    entries = load_entries(words=words, labels=labels, top_n=top_n)
    if not entries:
        print("[!] No entries matched your filter.")
        return

    by_sign = {}
    for e in entries:
        key = e["text"]
        by_sign.setdefault(key, []).append(e)

    print(f"\n[*] Signs to process : {sorted(by_sign.keys())}")
    print(f"[*] Max clips per sign: {max_per_sign}")
    print(f"[*] Parallel workers  : {workers}")

    stats = {}
    for sign_text, sign_entries in sorted(by_sign.items()):

        # TTS audio
        audio_dir  = Path("audio_signs")
        audio_dir.mkdir(exist_ok=True)
        audio_path = audio_dir / f"{sign_text}.mp3"
        if not audio_path.exists():
            try:
                gTTS(text=sign_text, lang='en', slow=False).save(str(audio_path))
                print(f"  [+] TTS: {audio_path}")
            except Exception as e:
                print(f"  [!] TTS failed: {e}")

        out_dir = Path(DATASET_PATH) / sign_text
        out_dir.mkdir(parents=True, exist_ok=True)

        existing    = [int(f.stem) for f in out_dir.glob("*.npy") if f.stem.isdigit()]
        start_idx   = max(existing) + 1 if existing else 0

        if start_idx >= max_per_sign:
            print(f"  [{sign_text}] Already has {start_idx} sequences, skipping.")
            stats[sign_text] = 0
            continue

        need = max_per_sign - start_idx
        print(f"\n{'='*52}")
        print(f"  Sign: {sign_text}  (have {start_idx}, need {need} more, ⚡ {workers} threads)")
        print(f"{'='*52}")

        # Shared mutable state (protected by lock)
        lock             = threading.Lock()
        seq_idx_holder   = [start_idx]      # list so lambda/closure can mutate it
        wrist_pos_list   = []
        gift_first_holder = [start_idx == 0]  # save preview only if no gif yet

        saved = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _process_one_clip,
                    entry, out_dir, seq_idx_holder, lock,
                    wrist_pos_list, gift_first_holder, sign_text, max_per_sign
                )
                for entry in sign_entries
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    if fut.result():
                        saved += 1
                except Exception as ex:
                    print(f"  [!] Worker error: {ex}")

        stats[sign_text] = saved

        # Update sign_zones.json
        if wrist_pos_list:
            zone  = compute_zone(wrist_pos_list)
            zones = load_sign_zones()
            zones[sign_text] = zone
            save_sign_zones(zones)
            print(f"  [📍] Zone '{sign_text}': x={zone['x']}, y={zone['y']}")

    # Summary
    print("\n" + "="*52)
    print(" DOWNLOAD SUMMARY")
    print("="*52)
    for sign, n in sorted(stats.items()):
        total = len([f for f in (Path(DATASET_PATH) / sign).glob("*.npy")])
        print(f"  {sign:<20} {n:>3} new   ({total} total)")

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
