import sys
import os
import time
import tempfile
import imageio
import cv2
import numpy as np
import requests
import speech_recognition as sr

try:
    from pose_format import Pose
    HAS_POSE = True
except ImportError:
    HAS_POSE = False

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QUrl, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap, QMovie
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QScrollArea, QProgressBar,
    QSizePolicy, QFrame, QMessageBox, QSlider
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaPlaylist, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

GIF_DIR     = "sign_gifs"
DATASET_DIR = "dataset"
MSASL_DIR   = "MS-ASL"                              # official (1000 signs)
VALID25_DIR = "MSASL-valid-dataset-downloader-main"  # pre-validated 25% (better URLs)

try:
    from asl_gloss import english_to_asl, asl_gloss_string
except ImportError:
    def english_to_asl(text): return text.lower().split()
    def asl_gloss_string(text): return text.upper()

# Accent matching 3_app_ui.py avatar border
ACCENT   = "#00d7ff"
BG_DARK  = "#1a1a1a"
BG_PANEL = "#2b2b2b"
BG_CARD  = "#222222"

# =============================================================================
# MS-ASL JSON Cache — loaded once at startup, used as an on-demand fallback
# =============================================================================
_MSASL_INDEX = None  # word → list of {url, start_time, end_time}

def _load_msasl_index():
    """Load both MS-ASL sources into a word → entries dict. valid-25% entries come first."""
    global _MSASL_INDEX
    if _MSASL_INDEX is not None:
        return _MSASL_INDEX
    import json
    index = {}

    def _ingest(fpath, text_key):
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for e in json.load(f):
                    if not isinstance(e, dict):
                        continue
                    word = (e.get(text_key) or e.get('text') or '').lower().strip()
                    if word:
                        entry = {
                            'url':        e.get('url', ''),
                            'start_time': float(e.get('start_time', 0) or 0),
                            'end_time':   float(e.get('end_time', 3) or 3),
                        }
                        index.setdefault(word, []).append(entry)
        except Exception:
            pass

    # Tier A: valid-25% first (pre-verified URLs, clean_text field)
    for fname in ('MSASL_TRAIN25.json', 'MSASL_VAL25.json', 'MASL_TEST25.json'):
        _ingest(os.path.join(VALID25_DIR, fname), 'clean_text')

    # Tier B: official MS-ASL for words not already covered
    for fname in ('MSASL_train.json', 'MSASL_val.json'):
        _ingest(os.path.join(MSASL_DIR, fname), 'text')

    _MSASL_INDEX = index
    return index


AI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_signs")
if not os.path.exists(AI_DIR):
    os.makedirs(AI_DIR)

# MediaPipe indexing for connections
# Pose: 0-32, L-Hand: 33-53 (offset 33), R-Hand: 54-74 (offset 54)
POSE_CO = [
    (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), # Torso & Arms
    (0,11), (0,12),                                                        # Head to Shoulders
    (23,25), (25,27), (24,26), (26,28),                                    # Legs (Upper/Lower)
    (15,33), (16,54)                                                       # WRIST BRIDGES (Pose to Hands)
]
HAND_CO = [
    (0,1), (1,2), (2,3), (3,4),    # Thumb
    (0,5), (5,6), (6,7), (7,8),    # Index
    (5,9), (9,10), (10,11), (11,12), # Middle
    (9,13), (13,14), (14,15), (15,16), # Ring
    (13,17), (17,18), (18,19), (19,20), # Pinky
    (0,17)                          # Palm base
]

def render_sign_mt_pose(text):
    if not HAS_POSE:
        print("[AI] pose-format not installed. Skipping.")
        return None
        
    out_path = os.path.join(AI_DIR, f"{text.replace(' ', '_')}_ai.mp4")
    if os.path.exists(out_path):
        return out_path

    print(f"[AI] Generating skeletal animation for '{text}'...")
    try:
        url = f"https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose?text={text}&spoken=en&signed=ase"
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"[AI] API Error: {resp.status_code}")
            return None

        p = Pose.read(resp.content)
        data = p.body.data
        if hasattr(data, 'filled'):
            data = data.filled(0)

        frames, persons, points, dims = data.shape
        w, h = 640, 480
        orig_w = p.header.dimensions.width or 512
        orig_h = p.header.dimensions.height or 512

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 25, (w, h))

        # Detect pose format offsets
        lh_off, rh_off = 33, 54
        curr_pose_co = POSE_CO
        if points == 178:
            lh_off, rh_off = 136, 157
            curr_pose_co = [
                (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
                (4, 136), (5, 157) # WRIST BRIDGES
            ]

        point_count = 0
        for f_idx in range(frames):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:] = (18, 18, 22) # Modern Charcoal background

            # Subtle Background Grid
            for gx in range(0, w, 40): cv2.line(img, (gx, 0), (gx, h), (25, 25, 30), 1)
            for gy in range(0, h, 40): cv2.line(img, (0, gy), (w, gy), (25, 25, 30), 1)

            # Floor shadow
            cv2.line(img, (40, 450), (600, 450), (45, 45, 50), 2, cv2.LINE_AA)

            pts = data[f_idx][0]

            def get_p(idx):
                nonlocal point_count
                if idx >= points: return None
                x, y, conf = pts[idx][:3]
                if conf <= 0.1: return None 
                scale = h / orig_h
                offset_x = (w - (orig_w * scale)) / 2
                px = int(x * scale + offset_x)
                py = int(y * scale)
                if 0 <= px < w and 0 <= py < h:
                    point_count += 1
                    return (px, py)
                return None

            # 1. Render Limbs
            for pair in curr_pose_co:
                p1, p2 = get_p(pair[0]), get_p(pair[1])
                if p1 and p2:
                    cv2.line(img, p1, p2, (180, 160, 0), 5, cv2.LINE_AA)
                    cv2.line(img, p1, p2, (255, 230, 0), 2, cv2.LINE_AA)

            # 2. Render Hands
            for pair in HAND_CO:
                p_l1, p_l2 = get_p(pair[0]+lh_off), get_p(pair[1]+lh_off)
                if p_l1 and p_l2:
                    cv2.line(img, p_l1, p_l2, (50, 127, 0), 3, cv2.LINE_AA)
                    cv2.line(img, p_l1, p_l2, (150, 255, 0), 1, cv2.LINE_AA)

                p_r1, p_r2 = get_p(pair[0]+rh_off), get_p(pair[1]+rh_off)
                if p_r1 and p_r2:
                    cv2.line(img, p_r1, p_r2, (127, 0, 75), 3, cv2.LINE_AA)
                    cv2.line(img, p_r1, p_r2, (255, 0, 150), 1, cv2.LINE_AA)

            # 3. Render Joints (Dotted Face)
            for pt_idx in range(points):
                p_coords = get_p(pt_idx)
                if p_coords:
                    color = (200, 200, 200); rad = 2
                    if points == 178:
                        if pt_idx < 8: color, rad = (0, 255, 255), 4 # Pose
                        elif pt_idx < 136: color, rad = (0, 150, 150), 1 # Dotted Face (Cyan-ish)
                        elif pt_idx < 157: color, rad = (0, 255, 100), 2 # LHand
                        else: color, rad = (200, 0, 255), 2 # RHand
                    else:
                        if pt_idx < 11: color, rad = (0, 255, 255), 2
                        elif pt_idx < 33: color, rad = (255, 200, 0), 3
                        elif pt_idx < 54: color, rad = (0, 255, 100), 2
                        else: color, rad = (200, 0, 255), 2
                    cv2.circle(img, p_coords, rad, color, -1, cv2.LINE_AA)

            # Labeling
            cv2.putText(img, f"SIGN: {text.upper()}", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (120, 120, 130), 1, cv2.LINE_AA)
            out.write(img)

        out.release()
        print(f"[AI] Rendered '{text}' with {point_count} total landmark instances.")
        return out_path
    except Exception as e:
        print(f"[AI] Render error: {e}")
        import traceback
        traceback.print_exc()
        return None
def _try_download_msasl_clip(word):
    """
    Try to download the first working MS-ASL clip for `word` (valid-25% first,
    then official MS-ASL). Saves to sign_gifs/<word>.mp4. Returns path or None.
    """
    index = _load_msasl_index()
    entries = index.get(word.lower(), [])
    if not entries:
        return None

    try:
        import yt_dlp
        from moviepy.editor import VideoFileClip
        import shutil
    except ImportError:
        return None

    os.makedirs(GIF_DIR, exist_ok=True)
    out_path = os.path.join(GIF_DIR, f"{word}.mp4")

    for entry in entries[:10]:
        url        = entry.get("url", "")
        start_time = float(entry.get("start_time", 0) or 0)
        end_time   = float(entry.get("end_time", start_time + 3) or start_time + 3)
        if not url.startswith("http"):
            url = "https://" + url

        tmp_dir  = tempfile.mkdtemp()
        tmp_file = os.path.join(tmp_dir, "raw.%(ext)s")
        try:
            ydl_opts = {
                "format": "best[ext=mp4][height<=480]/best[ext=mp4]/best[height<=480]/best",
                "outtmpl": tmp_file,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            raw = next(
                (os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith("raw")),
                None
            )
            if not raw:
                continue
            clip = VideoFileClip(raw).subclip(max(0, start_time - 0.3), end_time + 0.3)
            clip.write_videofile(out_path, audio=False, verbose=False, logger=None)
            clip.close()
            return out_path
        except Exception:
            continue
        finally:
            import shutil as _sh
            _sh.rmtree(tmp_dir, ignore_errors=True)

    return None



def _find_hand_active_segment(video_path, min_duration=1.5, sample_every=3):
    """
    Scan a video with MediaPipe Hands and return (start_sec, end_sec) of the
    LONGEST contiguous segment where hands are detected.
    Returns (None, None) if no hand activity found.

    sample_every: only process every Nth frame for speed (3 = 10fps on 30fps video).
    """
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    hand_timestamps = []  # seconds where a hand was detected

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_timestamps.append(frame_idx / fps)
            frame_idx += 1

    cap.release()

    if not hand_timestamps:
        return None, None

    # Find longest contiguous run of hand-active timestamps (gap ≤ 1s allowed)
    GAP_TOLERANCE = 1.0  # seconds — brief occlusions are OK
    best_start = hand_timestamps[0]
    best_end   = hand_timestamps[0]
    cur_start  = hand_timestamps[0]
    cur_end    = hand_timestamps[0]

    for t in hand_timestamps[1:]:
        if t - cur_end <= GAP_TOLERANCE:
            cur_end = t
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = t

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end

    duration = best_end - best_start
    if duration < min_duration:
        return None, None

    return best_start, best_end


def _try_youtube_asl_search(word):
    """
    Tier 4 fallback — single words ONLY.

    Phase 1: Collect candidate video IDs from up to 3 search queries (no download yet).
    Phase 2: Download each candidate to a temp file, scan with MediaPipe to measure
             the length of the hand-active window.
    Phase 3: Keep only the candidate with the LONGEST hand-active segment,
             clip to that window ± 0.5s, and save to sign_gifs/<word>.mp4.

    Slower than first-success but produces the clearest sign demonstration.
    Results are cached in sign_gifs/ so the cost is only paid once per word.
    """
    # ── Gate: only for single words ───────────────────────────────────────────
    if len(word.split()) > 1:
        return None

    try:
        import yt_dlp
        from moviepy.editor import VideoFileClip
        import shutil
    except ImportError:
        return None

    os.makedirs(GIF_DIR, exist_ok=True)
    out_path = os.path.join(GIF_DIR, f"{word}.mp4")

    # Instant cache hit
    if os.path.exists(out_path):
        return out_path

    queries = [
        f"{word} ASL sign language",
        f"how to sign {word} ASL",
        f"{word} American Sign Language",
    ]

    # ── Phase 1: collect candidate video URLs (metadata only, no download) ────
    print(f"  [🔍] Collecting YouTube candidates for '{word}'...")
    seen_ids  = set()
    candidates = []   # list of YouTube watch URLs

    info_opts = {
        "quiet":       True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat":  True,      # metadata only — very fast
        "match_filter": yt_dlp.utils.match_filter_func("duration < 180"),
    }
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        for query in queries:
            try:
                info = ydl.extract_info(f"ytsearch3:{query}", download=False)
                for entry in (info.get("entries") or []):
                    vid_id = entry.get("id") or entry.get("url", "")
                    if vid_id and vid_id not in seen_ids:
                        seen_ids.add(vid_id)
                        candidates.append(f"https://www.youtube.com/watch?v={vid_id}")
            except Exception:
                continue

    if not candidates:
        print(f"  [!] No YouTube candidates found for '{word}'")
        return None

    print(f"  [📋] {len(candidates)} unique candidates — scanning all for best hand activity...")

    # ── Phase 2: download each candidate, measure hand-active duration ────────
    best_raw      = None
    best_seg      = (0.0, 5.0)
    best_duration = -1.0
    tmp_dirs      = []

    for i, url in enumerate(candidates):
        tmp_dir  = tempfile.mkdtemp()
        tmp_dirs.append(tmp_dir)
        tmp_file = os.path.join(tmp_dir, f"cand{i}.%(ext)s")
        try:
            dl_opts = {
                "format":      "best[ext=mp4][height<=480]/best[ext=mp4]/best",
                "outtmpl":     tmp_file,
                "quiet":       True,
                "no_warnings": True,
                "noplaylist":  True,
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([url])

            raw = next(
                (os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                 if f.startswith(f"cand{i}")),
                None
            )
            if not raw or not os.path.exists(raw):
                continue

            seg_start, seg_end = _find_hand_active_segment(raw)
            MAX_CLIP = 12.0  # cap: even a 3-min tutorial has one sign per ~5-10s
            if seg_start is not None:
                # Clamp to 12s max — avoids picking a 60s general tutorial
                capped_dur = min(seg_end - seg_start, MAX_CLIP)
            else:
                capped_dur = 0.0
            hand_dur = capped_dur
            print(f"    Candidate {i+1}: hand-active {hand_dur:.1f}s  ({url.split('=')[-1]})")

            if hand_dur > best_duration:
                best_duration = hand_dur
                best_raw      = raw
                if seg_start is not None:
                    best_seg = (seg_start, seg_start + capped_dur)
                else:
                    best_seg = (0.0, 5.0)

        except Exception as ex:
            print(f"    Candidate {i+1}: failed — {ex}")
            continue

    # ── Phase 3: clip the winner and save ─────────────────────────────────────
    try:
        if best_raw and os.path.exists(best_raw):
            seg_start, seg_end = best_seg
            padding    = 0.5
            clip_start = max(0.0, seg_start - padding)
            clip_end   = seg_end + padding
            print(f"  [✂] Best candidate: {best_duration:.1f}s hand-active — "
                  f"clipping {clip_start:.1f}s → {clip_end:.1f}s")
            clip = VideoFileClip(best_raw).subclip(clip_start, clip_end)
            clip.write_videofile(out_path, audio=False, verbose=False, logger=None)
            clip.close()
            return out_path
        else:
            print(f"  [!] No usable YouTube clip found for '{word}'")
            return None
    except Exception as ex:
        print(f"  [!] Failed to clip best candidate: {ex}")
        return out_path if os.path.exists(out_path) else None
    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# 4. Word to Media Matcher
#    Chain: local GIF → local MP4 → MS-ASL download → YouTube ASL search → None
# =============================================================================
def match_gifs(text):
    """
    1. Convert English → ASL gloss tokens
    2. For each token: GIF → MP4 → MS-ASL → YouTube ASL → None
    """
    asl_tokens = english_to_asl(text)
    print(f"[ASL Gloss] {' '.join(t.upper() for t in asl_tokens)}")
    results = []
    i = 0
    while i < len(asl_tokens):
        matched = False
        for length in range(min(4, len(asl_tokens) - i), 0, -1):
            phrase = " ".join(asl_tokens[i:i+length])
            for candidate in [phrase, ''.join(c for c in phrase if c.isalnum() or c == ' ').strip()]:
                gif_path = os.path.join(GIF_DIR, f"{candidate}.gif")
                mp4_path = os.path.join(GIF_DIR, f"{candidate}.mp4")

                if os.path.exists(gif_path):
                    print(f"  [✓ GIF] '{candidate}' → {gif_path}")
                    results.append((phrase, gif_path))
                    i += length; matched = True; break
                elif os.path.exists(mp4_path):
                    print(f"  [✓ MP4] '{candidate}' → {mp4_path}")
                    results.append((phrase, mp4_path))
                    i += length; matched = True; break
                else:
                    print(f"  [?] '{candidate}' — trying MS-ASL...")
                    downloaded = _try_download_msasl_clip(candidate)
                    if downloaded:
                        print(f"  [✓ MS-ASL] '{candidate}' → {downloaded}")
                        results.append((phrase, downloaded))
                        i += length; matched = True; break

                    print(f"  [?] '{candidate}' — trying YouTube ASL search...")
                    yt_path = _try_youtube_asl_search(candidate)
                    if yt_path:
                        print(f"  [✓ YouTube] '{candidate}' → {yt_path}")
                        results.append((phrase, yt_path))
                        i += length; matched = True; break

                    print(f"  [✗] '{candidate}' — no match found anywhere")
            if matched:
                break
        if not matched:
            results.append((asl_tokens[i], None))
            i += 1
    print(f"[match_gifs] Final: {[(w, bool(p)) for w, p in results]}")
    return results


# =============================================================================
# 5. Match Worker — runs match_gifs() in a background thread with progress
# =============================================================================
class MatchWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    done_signal     = pyqtSignal(list, list) # (hybrid_results, skeleton_results)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        words    = english_to_asl(self.text)  # English → ASL gloss
        total    = max(len(words), 1)
        hybrid_results   = []
        skeleton_results = []
        
        i        = 0
        word_num = 0
        print(f"[MatchWorker] Dual-Mode Processing: {' '.join(w.upper() for w in words)}")
        
        while i < len(words):
            word_num += 1
            pct = int((word_num / total) * 90)
            
            # --- 1. SKELETON MODE (Always AI for the single word) ---
            word = words[i]
            self.progress_signal.emit(pct, f"🤖 [Skeleton] AI Pose for '{word}'...")
            sk_path = render_sign_mt_pose(word)
            skeleton_results.append((word, sk_path))
            
            # --- 2. HYBRID MODE (Local -> MS-ASL -> YouTube -> AI) ---
            matched = False
            for length in range(min(4, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i+length])
                candidate = ''.join(c for c in phrase if c.isalnum() or c == ' ').strip()
                
                gif_path = os.path.join(GIF_DIR, f"{candidate}.gif")
                mp4_path = os.path.join(GIF_DIR, f"{candidate}.mp4")

                # Tiers 1 & 2: Local
                if os.path.exists(gif_path):
                    hybrid_results.append((phrase, gif_path))
                    self.progress_signal.emit(pct, f"✓ Hybrid '{phrase}'")
                    matched = True; break
                elif os.path.exists(mp4_path):
                    hybrid_results.append((phrase, mp4_path))
                    self.progress_signal.emit(pct, f"✓ Hybrid '{phrase}'")
                    matched = True; break
                
                # Tier 3: MS-ASL
                dl_path = _try_download_msasl_clip(candidate)
                if dl_path:
                    hybrid_results.append((phrase, dl_path))
                    self.progress_signal.emit(pct, f"↓ Hybrid '{phrase}'")
                    matched = True; break
                
                # Tier 4: YouTube
                yt_path = _try_youtube_asl_search(candidate)
                if yt_path:
                    hybrid_results.append((phrase, yt_path))
                    self.progress_signal.emit(pct, f"▶ Hybrid '{phrase}'")
                    matched = True; break
                
                # Tier 5: AI Pose (Only if length == 1)
                if length == 1:
                    # In Hybrid mode, we can reuse the sk_path we just got for Skeleton mode
                    if sk_path:
                        hybrid_results.append((word, sk_path))
                        self.progress_signal.emit(pct, f"✨ Hybrid '{word}'")
                        matched = True; break

            if not matched:
                hybrid_results.append((words[i], None))
                self.progress_signal.emit(pct, f"✗ No hybrid sign for '{words[i]}'")
            
            i += 1 # We process word-by-word now for Dual Mode consistency
            
        self.progress_signal.emit(100, "Done.")
        self.done_signal.emit(hybrid_results, skeleton_results)


# =============================================================================
# 1. OpenCV Video Player Thread
# =============================================================================
class VideoPlayerThread(QThread):
    frame_signal    = pyqtSignal(QImage)
    duration_signal = pyqtSignal(int)
    position_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._path     = None
        self._playing  = False
        self._seek_to  = -1
        self._run_flag = False
        self._fps      = 25

    def load(self, path):
        self._path    = path
        self._playing = True
        self._seek_to = 0
        self._cap_reset = True
        if not self.isRunning():
            self._run_flag = True
            self.start()

    def play(self):   self._playing = True
    def pause(self):  self._playing = False
    def seek(self, f): self._seek_to = f

    def stop_thread(self):
        self._run_flag = False
        self._playing  = False
        self.wait()

    def run(self):
        cap = None
        self._cap_reset = False

        while self._run_flag:
            if getattr(self, '_cap_reset', False) and self._path:
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(self._path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._fps = cap.get(cv2.CAP_PROP_FPS) or 25
                self.duration_signal.emit(total)
                self._cap_reset = False

            if self._seek_to >= 0 and cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                self._seek_to = -1

            if self._playing and cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.position_signal.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                    self.frame_signal.emit(qt_img.copy())
                    time.sleep(1.0 / self._fps)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._playing = False
            else:
                time.sleep(0.05)

        if cap:
            cap.release()


# =============================================================================
# 2. Transcription Worker
# =============================================================================
class TranscribeWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    done_signal     = pyqtSignal(str)
    error_signal    = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        tmp_wav = ""
        try:
            self.progress_signal.emit(10, "Extracting audio from video...")
            import subprocess
            tmp_wav = tempfile.mktemp(suffix=".wav")

            # Try FFmpeg first for highly-optimized STT audio (16kHz Mono)
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", self.video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    tmp_wav
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Fallback to MoviePy if FFmpeg is not installed
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(self.video_path)
                if clip.audio is None:
                    self.error_signal.emit("No audio track found in the video.")
                    return
                clip.audio.write_audiofile(tmp_wav, verbose=False, logger=None)
                clip.close()

            if not os.path.exists(tmp_wav):
                self.error_signal.emit("Audio extraction failed entirely.")
                return

            # ----------------------------------------------------------------
            # Chunked transcription — Google STT free API limit is 60 seconds.
            # We split the audio into CHUNK_SEC-second windows and join results.
            # ----------------------------------------------------------------
            CHUNK_SEC = 55
            recognizer = sr.Recognizer()
            all_parts = []

            with sr.AudioFile(tmp_wav) as source:
                # Measure total duration by peeking at sample rate and frames
                total_frames  = source.DURATION   # seconds (float)
                num_chunks    = max(1, int(total_frames / CHUNK_SEC) + 1)

                self.progress_signal.emit(30, f"Transcribing {num_chunks} chunk(s) — please wait...")

                for chunk_idx in range(num_chunks):
                    offset = chunk_idx * CHUNK_SEC
                    if offset >= total_frames:
                        break

                    pct = 30 + int(((chunk_idx + 1) / num_chunks) * 60)
                    self.progress_signal.emit(pct, f"Transcribing chunk {chunk_idx+1}/{num_chunks} ({int(offset)}s – {int(offset+CHUNK_SEC)}s)...")

                    with sr.AudioFile(tmp_wav) as src:
                        audio_chunk = recognizer.record(src, offset=offset, duration=CHUNK_SEC)

                    try:
                        part = recognizer.recognize_google(audio_chunk)
                        all_parts.append(part)
                    except sr.UnknownValueError:
                        pass   # Silence or unclear — skip this chunk
                    except sr.RequestError as e:
                        self.error_signal.emit(f"Google STT error on chunk {chunk_idx+1}: {e}")
                        return

            if not all_parts:
                self.error_signal.emit("Could not clearly understand any speech in the video.")
                return

            full_text = " ".join(all_parts)
            self.progress_signal.emit(100, "Done.")
            self.done_signal.emit(full_text)

        except Exception as e:
            self.error_signal.emit(f"Error: {e}")
        finally:
            if os.path.exists(tmp_wav):
                try: os.remove(tmp_wav)
                except: pass


# =============================================================================
# 3. Export Worker
# =============================================================================
class ExportWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    done_signal     = pyqtSignal(str)
    error_signal    = pyqtSignal(str)

    def __init__(self, export_items, output_path, fps=12):
        super().__init__()
        self.export_items = export_items
        self.output_path = output_path
        self.fps         = fps

    def run(self):
        try:
            all_frames = []
            total = len(self.export_items)
            for i, (word, path) in enumerate(self.export_items):
                self.progress_signal.emit(int((i / total) * 90), f"Processing sign {i+1}/{total}...")
                if path and os.path.exists(path):
                    reader = imageio.get_reader(path)
                    for frame in reader:
                        rgb = np.array(frame)
                        if rgb.ndim == 3 and rgb.shape[2] == 4:
                            rgb = rgb[:, :, :3]
                        rgb = cv2.resize(rgb, (480, 360))
                        all_frames.append(rgb)
                    reader.close()
                else:
                    # Generate text frames for 1.5 seconds if no GIF is available
                    num_frames = int(self.fps * 1.5)
                    for _ in range(num_frames):
                        frame = np.zeros((360, 480, 3), dtype=np.uint8)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = f"[{word.upper()}]"
                        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
                        text_x = (480 - text_size[0]) // 2
                        text_y = (360 + text_size[1]) // 2
                        cv2.putText(frame, text, (text_x, text_y), font, 1.5, (0, 215, 255), 3, cv2.LINE_AA)
                        all_frames.append(frame)
            self.progress_signal.emit(95, "Writing output video...")
            imageio.mimsave(self.output_path, all_frames, fps=self.fps, macro_block_size=1)
            self.progress_signal.emit(100, "Export complete.")
            self.done_signal.emit(self.output_path)
        except Exception as e:
            self.error_signal.emit(f"Export failed: {e}")


# =============================================================================
# 5. Word Card Widget  — matches avatar style from 3_app_ui.py
# =============================================================================
class WordCard(QFrame):
    def __init__(self, word, media_path=None):
        super().__init__()
        has_media = media_path is not None
        self.setFixedSize(160, 200)
        border = ACCENT if has_media else "#3a3a3a"
        bg     = "#1a2a1a" if has_media else BG_CARD
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border-radius: 8px; border: 2px solid {border}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        if has_media and media_path.endswith('.mp4'):
            # Play MP4 using OpenCV frames + QTimer (avoids Windows codec issues)
            print(f"[WordCard] Loading MP4: {media_path}")
            self.frames = []
            cap = cv2.VideoCapture(media_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (144, 108))
                h, w, ch = rgb.shape
                # Use tobytes() to prevent the numpy buffer from being GC'd before the QPixmap is built
                qt_img = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
                self.frames.append(QPixmap.fromImage(qt_img))
            cap.release()
            print(f"[WordCard] Loaded {len(self.frames)} frames for '{word}'")

            self.current_frame = 0
            self.video_lbl = QLabel()
            self.video_lbl.setFixedSize(144, 108)
            self.video_lbl.setAlignment(Qt.AlignCenter)
            self.video_lbl.setStyleSheet("border: none; background: transparent;")
            layout.addWidget(self.video_lbl)
            
            if self.frames:
                self.video_lbl.setPixmap(self.frames[0])
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.next_frame)
                self.timer.start(50) # Approx 20 FPS looping
                
            self.movie = None
        else:
            # Play GIF using QMovie, or show fallback —
            gif_lbl = QLabel()
            gif_lbl.setFixedSize(144, 108)
            gif_lbl.setAlignment(Qt.AlignCenter)
            gif_lbl.setStyleSheet("border: none; background: transparent;")
            if has_media:
                self.movie = QMovie(media_path)
                self.movie.setScaledSize(QSize(144, 108))
                gif_lbl.setMovie(self.movie)
                self.movie.start()
            else:
                # No sign available — show the word text prominently on a dark tile
                self.movie = None
                txt_lbl = QLabel(word.upper())
                txt_lbl.setFixedSize(144, 108)
                txt_lbl.setAlignment(Qt.AlignCenter)
                txt_lbl.setWordWrap(True)
                txt_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
                txt_lbl.setStyleSheet(
                    "color: #00d7ff; background: #0a0a0a;"
                    "border-radius: 6px; border: none; padding: 4px;"
                )
                layout.addWidget(txt_lbl)
                return  # Skip the duplicate word_lbl / status_lbl below for text cards
            layout.addWidget(gif_lbl)

        word_lbl = QLabel(word)
        word_lbl.setAlignment(Qt.AlignCenter)
        word_lbl.setWordWrap(True)
        word_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        word_lbl.setStyleSheet("color: white; border: none;")
        layout.addWidget(word_lbl)

        status_lbl = QLabel("matched" if has_media else "text only")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setFont(QFont("Segoe UI", 8))
        status_lbl.setStyleSheet(f"color: {ACCENT if has_media else '#666'}; border: none;")
        layout.addWidget(status_lbl)

    def next_frame(self):
        if hasattr(self, 'frames') and self.frames:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.video_lbl.setPixmap(self.frames[self.current_frame])


# =============================================================================
# 6. Main Window — styled to match 3_app_ui.py
# =============================================================================
class TranslatorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            f"background-color: {BG_DARK}; color: white; font-family: Segoe UI, sans-serif;"
        )

        self.video_path           = None
        self.hybrid_items         = []
        self.skeleton_items       = []
        self._total_frames        = 1
        self._is_playing          = False
        self._view_mode           = "hybrid" # or "skeleton"

        self.player_thread = VideoPlayerThread()
        self.player_thread.frame_signal.connect(self.update_video_frame)
        self.player_thread.duration_signal.connect(self.on_duration)
        self.player_thread.position_signal.connect(self.on_position)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # Header
        header = QLabel("Video to Sign Language Translator")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        root.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {ACCENT};")
        root.addWidget(sep)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet(
            f"background: {BG_PANEL}; padding: 8px 12px; border-radius: 8px; color: #aaaaaa;"
        )
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ctrl.addWidget(self.file_label)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedHeight(34)
        browse_btn.setStyleSheet(self._btn("#0057d9", "#0046b0"))
        browse_btn.clicked.connect(self.browse_file)
        ctrl.addWidget(browse_btn)

        self.translate_btn = QPushButton("Translate")
        self.translate_btn.setFixedHeight(34)
        self.translate_btn.setStyleSheet(self._btn("#00882b", "#006622"))
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self.start_translation)
        ctrl.addWidget(self.translate_btn)

        self.export_hybrid_btn = QPushButton("Export Hybrid")
        self.export_hybrid_btn.setFixedHeight(34)
        self.export_hybrid_btn.setStyleSheet(self._btn("#7a3d00", "#5c2e00"))
        self.export_hybrid_btn.setEnabled(False)
        self.export_hybrid_btn.clicked.connect(lambda: self.export_video("hybrid"))
        ctrl.addWidget(self.export_hybrid_btn)

        self.export_skel_btn = QPushButton("Export Skeleton")
        self.export_skel_btn.setFixedHeight(34)
        self.export_skel_btn.setStyleSheet(self._btn("#4a148c", "#311b92"))
        self.export_skel_btn.setEnabled(False)
        self.export_skel_btn.clicked.connect(lambda: self.export_video("skeleton"))
        ctrl.addWidget(self.export_skel_btn)

        root.addLayout(ctrl)

        # Progress
        self.progress = QProgressBar()
        self.progress.setFixedHeight(5)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet(f"""
            QProgressBar {{ background: #333333; border-radius: 3px; border: none; }}
            QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
        """)
        root.addWidget(self.progress)

        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(self.status_lbl)

        # Main split: left = video | right = sign cards
        main_area = QHBoxLayout()
        main_area.setSpacing(16)

        # LEFT — video preview
        left = QVBoxLayout()
        left.setSpacing(8)

        self.video_label = QLabel("Select a video to preview")
        self.video_label.setMinimumSize(480, 300)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background: #000000; border-radius: 10px; color: #444444; font-size: 13px;"
        )
        left.addWidget(self.video_label, stretch=1)

        # Transport controls
        transport = QHBoxLayout()
        transport.setSpacing(8)

        self.play_btn = QPushButton("Play")
        self.play_btn.setFixedHeight(30)
        self.play_btn.setFixedWidth(60)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(self._btn("#333333", "#444444"))
        self.play_btn.clicked.connect(self.toggle_play)
        transport.addWidget(self.play_btn)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.setValue(0)
        self.seek_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background: #333333; height: 4px; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; height: 12px;
                                          margin: -4px 0; border-radius: 6px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
        """)
        self.seek_slider.sliderMoved.connect(self.on_seek)
        transport.addWidget(self.seek_slider, stretch=1)

        self.time_lbl = QLabel("0:00 / 0:00")
        self.time_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        transport.addWidget(self.time_lbl)
        left.addLayout(transport)

        # Transcript box — styled like the system bubble in 3_app_ui.py
        self.transcript_lbl = QLabel("Transcript will appear here after translation.")
        self.transcript_lbl.setWordWrap(True)
        self.transcript_lbl.setAlignment(Qt.AlignCenter)
        self.transcript_lbl.setFixedHeight(55)
        self.transcript_lbl.setFont(QFont("Segoe UI", 11))
        self.transcript_lbl.setStyleSheet(
            f"background: {BG_PANEL}; color: #cccccc; padding: 8px; border-radius: 10px;"
        )
        left.addWidget(self.transcript_lbl)
        main_area.addLayout(left, stretch=5)

        # RIGHT — sign card panel (matches right panel from 3_app_ui.py)
        right_panel = QWidget()
        right_panel.setStyleSheet(
            f"background-color: {BG_PANEL}; border-radius: 15px;"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        panel_header = QLabel("Sign Translations")
        panel_header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        panel_header.setAlignment(Qt.AlignCenter)
        panel_header.setStyleSheet(f"color: white;")
        right_layout.addWidget(panel_header)

        # View Toggle
        toggle_box = QHBoxLayout()
        toggle_box.setContentsMargins(10, 0, 10, 0)
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self.view_group = QButtonGroup(self)
        
        self.radio_hybrid = QRadioButton("Hybrid (Real+AI)")
        self.radio_hybrid.setChecked(True)
        self.radio_hybrid.setStyleSheet("color: white; font-size: 11px;")
        self.radio_hybrid.toggled.connect(lambda: self.switch_view("hybrid"))
        self.view_group.addButton(self.radio_hybrid)
        toggle_box.addWidget(self.radio_hybrid)

        self.radio_skel = QRadioButton("100% Skeleton (AI)")
        self.radio_skel.setStyleSheet("color: white; font-size: 11px;")
        self.radio_skel.toggled.connect(lambda: self.switch_view("skeleton"))
        self.view_group.addButton(self.radio_skel)
        toggle_box.addWidget(self.radio_skel)
        
        right_layout.addLayout(toggle_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_flow = QHBoxLayout(self.cards_container)
        self.cards_flow.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.cards_flow.setSpacing(10)
        self.cards_flow.setContentsMargins(6, 6, 6, 6)
        scroll.setWidget(self.cards_container)
        right_layout.addWidget(scroll)

        main_area.addWidget(right_panel, stretch=4)
        root.addLayout(main_area, stretch=1)

    # ------- Style helper -------
    def _btn(self, color, hover):
        return f"""
            QPushButton {{
                background-color: {color}; color: white;
                border-radius: 7px; padding: 0 14px;
                font-size: 12px; font-weight: bold; border: none;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #3a3a3a; color: #666666; }}
        """

    @staticmethod
    def _frames_to_str(frames, fps):
        s = int(frames / max(fps, 1))
        return f"{s // 60}:{s % 60:02d}"

    # ------- File browsing -------
    def browse_file(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
                options=QFileDialog.DontUseNativeDialog
            )
            if not path:
                return
            self.video_path = path
            self.file_label.setText(os.path.basename(path))
            self.file_label.setStyleSheet(
                f"background: {BG_PANEL}; padding: 8px 12px; border-radius: 8px; color: white;"
            )
            self.translate_btn.setEnabled(True)
            self._clear_cards()
            self.transcript_lbl.setText("Transcript will appear here after translation.")
            self.video_label.setStyleSheet("background: #000000; border-radius: 10px;")
            self.play_btn.setEnabled(True)
            self._is_playing = True
            self.play_btn.setText("Pause")
            self.player_thread.load(path)
            
            # Reset results
            self.hybrid_items = []
            self.skeleton_items = []
            self.export_hybrid_btn.setEnabled(False)
            self.export_skel_btn.setEnabled(False)
        except Exception as e:
            print(f"[ERROR] browse_file failed: {e}")
            QMessageBox.critical(self, "Browse Error", f"Could not load video: {e}")

    def update_video_frame(self, qt_img):
        scaled = qt_img.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def on_duration(self, total_frames):
        self._total_frames = max(total_frames, 1)
        self.seek_slider.setRange(0, total_frames)

    def on_position(self, frame_idx):
        self.seek_slider.setValue(frame_idx)
        fps = self.player_thread._fps
        self.time_lbl.setText(
            f"{self._frames_to_str(frame_idx, fps)} / "
            f"{self._frames_to_str(self._total_frames, fps)}"
        )

    def toggle_play(self):
        if self._is_playing:
            self._is_playing = False
            self.player_thread.pause()
            self.play_btn.setText("Play")
        else:
            self._is_playing = True
            self.player_thread.play()
            self.play_btn.setText("Pause")

    def on_seek(self, value):
        self.player_thread.seek(value)

    # ------- Translation -------
    def start_translation(self):
        if not self.video_path:
            return
        self._clear_cards()
        self.transcript_lbl.setText("Transcribing...")
        self.translate_btn.setEnabled(False)
        self.export_hybrid_btn.setEnabled(False)
        self.export_skel_btn.setEnabled(False)
        self.progress.setValue(0)

        self.worker = TranscribeWorker(self.video_path)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.done_signal.connect(self.on_transcript_ready)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.status_lbl.setText(msg)

    def on_transcript_ready(self, text):
        self.transcript_lbl.setText(f'"{text}"')
        self._clear_cards()

        # Show an animated waiting message
        self._waiting_lbl = QLabel("Matching signs (Dual-Mode)...")
        self._waiting_lbl.setAlignment(Qt.AlignCenter)
        self._waiting_lbl.setFont(QFont("Segoe UI", 12))
        self._waiting_lbl.setStyleSheet(f"color: {ACCENT}; background: transparent; border: none;")
        self.cards_flow.addWidget(self._waiting_lbl)

        # Animated dots every 400ms
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._pulse_waiting)
        self._dot_timer.start(400)

        # Run the full matching in a background thread
        self.match_worker = MatchWorker(text)
        self.match_worker.progress_signal.connect(self.on_progress)
        self.match_worker.done_signal.connect(self.on_match_done)
        self.match_worker.start()

    def _pulse_waiting(self):
        if not hasattr(self, "_waiting_lbl") or not self._waiting_lbl:
            return
        self._dot_count = (self._dot_count + 1) % 4
        dots = '.' * self._dot_count
        pct = self.progress.value()
        try:
            self._waiting_lbl.setText(f"Matching signs{dots}  ({pct}%)")
        except:
            pass # Handle case where widget might be deleted mid-pulse

    def on_match_done(self, hybrid, skeleton):
        if hasattr(self, "_dot_timer"):
            self._dot_timer.stop()
        self._clear_cards()

        self.hybrid_items = hybrid
        self.skeleton_items = skeleton
        self.translate_btn.setEnabled(True)
        self.export_hybrid_btn.setEnabled(True)
        self.export_skel_btn.setEnabled(True)
        
        self.switch_view(self._view_mode)
        self.status_lbl.setText("Translation complete. Toggle view to see Full Skeleton version.")

    def switch_view(self, mode):
        self._view_mode = mode
        self._clear_cards()
        items = self.hybrid_items if mode == "hybrid" else self.skeleton_items
        for word, path in items:
            self.cards_flow.addWidget(WordCard(word, path))

    def on_error(self, msg):
        self.progress.setValue(0)
        self.status_lbl.setText(msg)
        self.translate_btn.setEnabled(True)
        QMessageBox.warning(self, "Error", msg)

    # ------- Export -------
    def export_video(self, mode="hybrid"):
        items = self.hybrid_items if mode == "hybrid" else self.skeleton_items
        if not items: return
        
        fname = f"translated_{mode}_{int(time.time())}.mp4"
        out, _ = QFileDialog.getSaveFileName(self, f"Save {mode.capitalize()} Video", fname, "MP4 Video (*.mp4)")
        if not out: return
        
        self.export_hybrid_btn.setEnabled(False)
        self.export_skel_btn.setEnabled(False)
        self.translate_btn.setEnabled(False)
        
        self.status_lbl.setText(f"Exporting {mode} video...")
        self.export_worker = ExportWorker(items, out)
        self.export_worker.progress_signal.connect(self.on_progress)
        self.export_worker.error_signal.connect(self.on_error)
        self.export_worker.done_signal.connect(lambda p: (
            QMessageBox.information(self, "Export", f"Video saved to: {p}"),
            self.export_hybrid_btn.setEnabled(True),
            self.export_skel_btn.setEnabled(True),
            self.translate_btn.setEnabled(True)
        ))
        self.export_worker.start()

    def on_export_done(self, path):
        self.export_btn.setEnabled(True)
        self.translate_btn.setEnabled(True)
        self.status_lbl.setText(f"Saved to: {path}")
        QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")

    def _clear_cards(self):
        while self.cards_flow.count():
            item = self.cards_flow.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.export_items = []

    def closeEvent(self, event):
        if hasattr(self, 'player_thread'): self.player_thread.stop_thread()
        event.accept()

class VideoTranslateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("English to ASL Speech/Video Translator")
        self.resize(1100, 800)
        self.widget = TranslatorWidget()
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        self.widget.closeEvent(event)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoTranslateApp()
    win.show()
    sys.exit(app.exec_())
