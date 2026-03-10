import sys
import os
import time
import tempfile
import imageio
import cv2
import numpy as np
import speech_recognition as sr

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
MSASL_DIR   = r"C:\Users\AMEER AKBAR\Downloads\MS-ASL\MS-ASL"

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
    """Load MS-ASL JSON(s) into a word → entries dict. Returns {} on failure."""
    global _MSASL_INDEX
    if _MSASL_INDEX is not None:
        return _MSASL_INDEX
    import json
    index = {}
    for fname in ("MSASL_train.json", "MSASL_val.json"):
        fpath = os.path.join(MSASL_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                entries = json.load(f)
            for e in entries:
                word = e.get("text", "").lower().strip()
                if word:
                    index.setdefault(word, []).append(e)
        except Exception:
            pass
    _MSASL_INDEX = index
    return index


def _try_download_msasl_clip(word):
    """
    Try to download the first working MS-ASL YouTube clip for `word`
    and save it as sign_gifs/<word>.mp4. Returns the path on success, None on failure.
    """
    index = _load_msasl_index()
    entries = index.get(word.lower(), [])
    if not entries:
        return None

    try:
        import yt_dlp
        from moviepy.editor import VideoFileClip
        import tempfile, shutil
    except ImportError:
        return None

    os.makedirs(GIF_DIR, exist_ok=True)
    out_path = os.path.join(GIF_DIR, f"{word}.mp4")

    for entry in entries[:10]:  # Try up to 10 entries per word
        url        = entry.get("url", "")
        start_time = entry.get("start_time", 0)
        end_time   = entry.get("end_time", start_time + 3)
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
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return None


# =============================================================================
# 4. Word to Media Matcher (GIF → sign_gifs MP4 → MS-ASL on-demand → None)
# =============================================================================
def match_gifs(text):
    words = text.lower().split()
    print(f"[match_gifs] Transcript words: {words}")
    results = []
    i = 0
    while i < len(words):
        matched = False
        for length in range(min(4, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i+length])
            for candidate in [phrase, ''.join(c for c in phrase if c.isalnum() or c == ' ').strip()]:
                # Priority 1: sign_gifs/<word>.gif (from manual GIF collection)
                gif_path = os.path.join(GIF_DIR, f"{candidate}.gif")
                # Priority 2: sign_gifs/<word>.mp4 (saved by 7_msasl_downloader)
                mp4_path = os.path.join(GIF_DIR, f"{candidate}.mp4")

                if os.path.exists(gif_path):
                    print(f"  [✓ GIF] '{candidate}' → {gif_path}")
                    results.append((phrase, gif_path))
                    i += length
                    matched = True
                    break
                elif os.path.exists(mp4_path):
                    print(f"  [✓ MP4] '{candidate}' → {mp4_path}")
                    results.append((phrase, mp4_path))
                    i += length
                    matched = True
                    break
                else:
                    print(f"  [?] '{candidate}' not in sign_gifs/ — trying MS-ASL...")
                    # Priority 3: Try on-demand download from MS-ASL JSON
                    downloaded = _try_download_msasl_clip(candidate)
                    if downloaded:
                        print(f"  [✓ DL]  '{candidate}' downloaded → {downloaded}")
                        results.append((phrase, downloaded))
                        i += length
                        matched = True
                        break
                    else:
                        print(f"  [✗]    '{candidate}' — no match found anywhere")
            if matched:
                break
        if not matched:
            results.append((words[i], None))
            i += 1
    print(f"[match_gifs] Final results: {[(w, bool(p)) for w, p in results]}")
    return results


# =============================================================================
# 5. Match Worker — runs match_gifs() in a background thread with progress
# =============================================================================
class MatchWorker(QThread):
    progress_signal = pyqtSignal(int, str)    # (pct, status_msg)
    done_signal     = pyqtSignal(list)        # full results list when complete

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        words = self.text.lower().split()
        total = max(len(words), 1)
        results = []
        i = 0
        word_num = 0
        while i < len(words):
            matched = False
            word_num += 1
            pct = int((word_num / total) * 90)
            for length in range(min(4, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i+length])
                for candidate in [phrase, ''.join(c for c in phrase if c.isalnum() or c == ' ').strip()]:
                    gif_path = os.path.join(GIF_DIR, f"{candidate}.gif")
                    mp4_path = os.path.join(GIF_DIR, f"{candidate}.mp4")
                    if os.path.exists(gif_path):
                        results.append((phrase, gif_path))
                        self.progress_signal.emit(pct, f"Matched '{phrase}' ✓")
                        i += length; matched = True; break
                    elif os.path.exists(mp4_path):
                        results.append((phrase, mp4_path))
                        self.progress_signal.emit(pct, f"Matched '{phrase}' ✓")
                        i += length; matched = True; break
                    else:
                        self.progress_signal.emit(pct, f"Searching MS-ASL for '{candidate}'...")
                        downloaded = _try_download_msasl_clip(candidate)
                        if downloaded:
                            results.append((phrase, downloaded))
                            self.progress_signal.emit(pct, f"Downloaded '{phrase}' ↓")
                            i += length; matched = True; break
                if matched:
                    break
            if not matched:
                results.append((words[i], None))
                self.progress_signal.emit(pct, f"No sign for '{words[i]}'")
                i += 1
        self.progress_signal.emit(100, "Done.")
        self.done_signal.emit(results)


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
class VideoTranslateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Video Translator")
        self.resize(1200, 750)
        self.setStyleSheet(
            f"background-color: {BG_DARK}; color: white; font-family: Segoe UI, sans-serif;"
        )

        self.video_path           = None
        self.export_items         = []
        self._total_frames        = 1
        self._is_playing          = False

        self.player_thread = VideoPlayerThread()
        self.player_thread.frame_signal.connect(self.update_video_frame)
        self.player_thread.duration_signal.connect(self.on_duration)
        self.player_thread.position_signal.connect(self.on_position)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
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

        self.export_btn = QPushButton("Export Video")
        self.export_btn.setFixedHeight(34)
        self.export_btn.setStyleSheet(self._btn("#7a3d00", "#5c2e00"))
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_video)
        ctrl.addWidget(self.export_btn)

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if not path:
            return
        self.video_path = path
        self.file_label.setText(os.path.basename(path))
        self.file_label.setStyleSheet(
            f"background: {BG_PANEL}; padding: 8px 12px; border-radius: 8px; color: white;"
        )
        self.translate_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self._clear_cards()
        self.transcript_lbl.setText("Transcript will appear here after translation.")
        self.video_label.setStyleSheet("background: #000000; border-radius: 10px;")
        self.play_btn.setEnabled(True)
        self._is_playing = True
        self.play_btn.setText("Pause")
        self.player_thread.load(path)

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
        self.export_btn.setEnabled(False)
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

        # Show an animated waiting message while matching runs in a background thread
        self._waiting_lbl = QLabel("⏳  Matching signs...")
        self._waiting_lbl.setAlignment(Qt.AlignCenter)
        self._waiting_lbl.setFont(QFont("Segoe UI", 12))
        self._waiting_lbl.setStyleSheet(
            f"color: {ACCENT}; background: transparent; border: none;"
        )
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
        self._dot_count = (self._dot_count + 1) % 4
        dots = '.' * self._dot_count
        pct = self.progress.value()
        self._waiting_lbl.setText(f"⏳  Matching signs{dots}  ({pct}%)")

    def on_match_done(self, matched):
        self._dot_timer.stop()
        self._clear_cards()

        self.export_items = matched
        for word, gif_path in matched:
            self.cards_flow.addWidget(WordCard(word, gif_path))
        self.translate_btn.setEnabled(True)
        if self.export_items:
            self.export_btn.setEnabled(True)
        num_matched = len([p for w, p in matched if p])
        self.status_lbl.setText(
            f"{num_matched} of {len(matched)} words matched to signs."
        )


    def on_error(self, msg):
        self.progress.setValue(0)
        self.status_lbl.setText(msg)
        self.translate_btn.setEnabled(True)
        QMessageBox.warning(self, "Error", msg)

    # ------- Export -------
    def export_video(self):
        out, _ = QFileDialog.getSaveFileName(
            self, "Save Sign Video", "sign_translation.mp4", "MP4 Video (*.mp4)"
        )
        if not out:
            return
        self.export_btn.setEnabled(False)
        self.translate_btn.setEnabled(False)
        self.export_worker = ExportWorker(self.export_items, out)
        self.export_worker.progress_signal.connect(self.on_progress)
        self.export_worker.error_signal.connect(self.on_error)
        self.export_worker.done_signal.connect(self.on_export_done)
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
        self.player_thread.stop_thread()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoTranslateApp()
    win.show()
    sys.exit(app.exec_())
