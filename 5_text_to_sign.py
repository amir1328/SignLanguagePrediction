import sys
import os
import cv2
import numpy as np
import importlib
import imageio

from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont, QMovie, QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QSizePolicy,
    QTextEdit, QMainWindow, QFileDialog, QProgressBar, QMessageBox
)

GIF_DIR = 'sign_gifs'

try:
    from asl_gloss import english_to_asl, asl_gloss_string
except ImportError:
    def english_to_asl(text): return text.lower().split()
    def asl_gloss_string(text): return text.upper()

ACCENT   = '#00d7ff'
BG_DARK  = '#1a1a1a'
BG_PANEL = '#2b2b2b'
BG_CARD  = '#222222'

def match_gifs(text):
    asl_tokens = english_to_asl(text)
    results = []
    try:
        vt = importlib.import_module('4_video_translate')
        render_ai = vt.render_sign_mt_pose
    except:
        render_ai = None

    i = 0
    while i < len(asl_tokens):
        matched = False
        word_to_use = asl_tokens[i]
        hybrid_path = None
        ai_path = None

        for length in range(min(4, len(asl_tokens) - i), 1, -1):
            phrase = ' '.join(asl_tokens[i:i+length])
            candidate = ''.join(c for c in phrase if c.isalnum() or c == ' ').strip()
            g_p = os.path.join(GIF_DIR, f'{candidate}.gif')
            m_p = os.path.join(GIF_DIR, f'{candidate}.mp4')
            if os.path.exists(g_p):
                hybrid_path = g_p; word_to_use = phrase; i += length; matched = True; break
            elif os.path.exists(m_p):
                hybrid_path = m_p; word_to_use = phrase; i += length; matched = True; break
        
        if not matched:
            word = asl_tokens[i]
            candidate = ''.join(c for c in word if c.isalnum()).strip()
            g_p = os.path.join(GIF_DIR, f'{candidate}.gif')
            m_p = os.path.join(GIF_DIR, f'{candidate}.mp4')
            if os.path.exists(g_p): hybrid_path = g_p
            elif os.path.exists(m_p): hybrid_path = m_p
            i += 1

        if render_ai:
            ai_path = render_ai(word_to_use)
            
        results.append((word_to_use, hybrid_path, ai_path))
    return results

class ExportWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    done_signal     = pyqtSignal(str)
    error_signal    = pyqtSignal(str)

    def __init__(self, export_items, output_path, fps=15):
        super().__init__()
        self.export_items = export_items
        self.output_path = output_path
        self.fps         = fps

    def run(self):
        try:
            all_frames = []
            total = len(self.export_items)
            for i, (word, path) in enumerate(self.export_items):
                self.progress_signal.emit(int((i / total) * 90), f"Processing {word}...")
                if path and os.path.exists(path):
                    reader = imageio.get_reader(path)
                    for frame in reader:
                        rgb = np.array(frame)
                        if rgb.ndim == 3 and rgb.shape[2] == 4:
                            rgb = rgb[:, :, :3]
                        rgb = cv2.resize(rgb, (640, 480))
                        all_frames.append(rgb)
                    reader.close()
                else:
                    for _ in range(int(self.fps * 1.2)):
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, f"[{word.upper()}]", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 3)
                        all_frames.append(frame)
            
            if not all_frames: raise Exception("No frames generated")
            imageio.mimsave(self.output_path, all_frames, fps=self.fps, macro_block_size=1)
            self.done_signal.emit(self.output_path)
        except Exception as e:
            self.error_signal.emit(str(e))

class MediaLabel(QLabel):
    def __init__(self, path=None, label_text="Hybrid"):
        super().__init__()
        self.setFixedSize(140, 105)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #000; border-radius: 4px; border: 1px solid #333;")
        self.movie = None
        self.frames = []
        self.timer = None
        if path:
            if path.lower().endswith('.mp4'):
                self.load_mp4(path)
            else:
                self.movie = QMovie(path)
                self.movie.setScaledSize(QSize(140, 105))
                self.setMovie(self.movie)
                self.movie.start()
        else:
            self.setText(f"No {label_text}")
            self.setStyleSheet("color: #444; background: #050505; border-radius: 4px;")

    def load_mp4(self, path):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (140, 105))
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            self.frames.append(QPixmap.fromImage(qt_img))
        cap.release()
        if self.frames:
            self.current_frame = 0
            self.setPixmap(self.frames[0])
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_frame)
            self.timer.start(40)

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.setPixmap(self.frames[self.current_frame])

class WordCard(QFrame):
    def __init__(self, word, hybrid_path=None, ai_path=None):
        super().__init__()
        self.setFixedWidth(320)
        self.setFixedHeight(220)
        border = ACCENT if (hybrid_path or ai_path) else '#333'
        self.setStyleSheet(f'QFrame {{ background-color: {BG_CARD}; border-radius: 10px; border: 2px solid {border}; }}')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        media_row = QHBoxLayout()
        media_row.addWidget(MediaLabel(hybrid_path, "Hybrid"))
        media_row.addWidget(MediaLabel(ai_path, "AI Skeleton"))
        layout.addLayout(media_row)
        lbl_row = QHBoxLayout()
        for t in ["HYBRID", "AI SKELETON"]:
            l = QLabel(t); l.setAlignment(Qt.AlignCenter); l.setStyleSheet("color: #888; font-size: 9px; border: none;")
            lbl_row.addWidget(l)
        layout.addLayout(lbl_row)
        word_lbl = QLabel(word.upper())
        word_lbl.setAlignment(Qt.AlignCenter)
        word_lbl.setFont(QFont('Segoe UI', 11, QFont.Bold))
        word_lbl.setStyleSheet('color: white; border: none; margin-top: 5px;')
        layout.addWidget(word_lbl)

class TextTranslateWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.results = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        header = QLabel('TEXT TO SIGN TRANSLATOR'); header.setFont(QFont('Segoe UI', 18, QFont.Bold)); header.setStyleSheet(f'color: {ACCENT};')
        layout.addWidget(header)
        self.input_box = QTextEdit(); self.input_box.setPlaceholderText('Type here...'); self.input_box.setFixedHeight(80)
        self.input_box.setStyleSheet('background: #111; color: white; border: 1px solid #333; border-radius: 8px; padding: 10px;')
        layout.addWidget(self.input_box)
        btn_lay = QHBoxLayout()
        self.trans_btn = QPushButton('Translate'); self.trans_btn.setStyleSheet(f'background: {ACCENT}; color: black; font-weight: bold; height: 35px;')
        self.trans_btn.clicked.connect(self.translate); btn_lay.addWidget(self.trans_btn)
        self.exp_h_btn = QPushButton('Export Hybrid'); self.exp_h_btn.setEnabled(False); self.exp_h_btn.clicked.connect(lambda: self.export_signs("hybrid"))
        self.exp_a_btn = QPushButton('Export AI'); self.exp_a_btn.setEnabled(False); self.exp_a_btn.clicked.connect(lambda: self.export_signs("ai"))
        btn_lay.addWidget(self.exp_h_btn); btn_lay.addWidget(self.exp_a_btn)
        layout.addLayout(btn_lay)
        self.progress = QProgressBar(); self.progress.setVisible(False); layout.addWidget(self.progress)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet('background: transparent; border: none;')
        self.container = QWidget(); self.flow_layout = QVBoxLayout(self.container); scroll.setWidget(self.container); layout.addWidget(scroll)

    def translate(self):
        text = self.input_box.toPlainText().strip()
        if not text: return
        while self.flow_layout.count():
            item = self.flow_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.results = match_gifs(text)
        for w, h, a in self.results: self.flow_layout.addWidget(WordCard(w, h, a))
        self.exp_h_btn.setEnabled(True); self.exp_a_btn.setEnabled(True)

    def export_signs(self, mode):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", f"signs_{mode}.mp4", "MP4 Video (*.mp4)")
        if not path: return
        items = [(w, h if mode=="hybrid" else a) for w, h, a in self.results]
        self.worker = ExportWorker(items, path)
        self.worker.progress_signal.connect(lambda p, s: self.progress.setValue(p))
        self.worker.done_signal.connect(lambda f: QMessageBox.information(self, "Done", f"Video saved to {f}"))
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.progress.setVisible(True); self.worker.start()
