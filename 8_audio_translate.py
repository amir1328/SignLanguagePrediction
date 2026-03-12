import sys
import os
import threading
import queue
import importlib
import speech_recognition as sr
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QMovie, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QSizePolicy,
    QMainWindow, QFileDialog, QProgressBar, QMessageBox
)

# Use importlib for numeric filename
try:
    text_to_sign = importlib.import_module('5_text_to_sign')
    WordCard = text_to_sign.WordCard
    match_gifs = text_to_sign.match_gifs
    ExportWorker = text_to_sign.ExportWorker
    ACCENT = text_to_sign.ACCENT
except Exception as e:
    print(f"Error importing 5_text_to_sign: {e}")
    class WordCard(QFrame): pass
    def match_gifs(t): return []
    class ExportWorker(QThread): pass
    ACCENT = "#00d7ff"

class AudioWorker(QThread):
    heard_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, mode="mic", file_path=None):
        super().__init__()
        self._run_flag = True
        self.recognizer = sr.Recognizer()
        self.mode = mode
        self.file_path = file_path

    def run(self):
        if self.mode == "file":
            self.process_file()
        else:
            self.process_mic()

    def process_file(self):
        if not self.file_path: return
        self.status_signal.emit(f"Processing File: {os.path.basename(self.file_path)}...")
        try:
            with sr.AudioFile(self.file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                if text:
                    self.heard_signal.emit(text)
                self.status_signal.emit("File processed.")
        except Exception as e:
            self.status_signal.emit(f"File Error: {e}")

    def process_mic(self):
        try:
            mic = sr.Microphone()
        except Exception as e:
            self.status_signal.emit(f'Microphone Error: {e}')
            return

        with mic as source:
            self.status_signal.emit('Calibrating noise...')
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            self.status_signal.emit('Listening...')

            while self._run_flag:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    self.status_signal.emit('Processing...')
                    text = self.recognizer.recognize_google(audio)
                    if text:
                        self.heard_signal.emit(text)
                    self.status_signal.emit('Listening...')
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.status_signal.emit('Listening...')
                    continue
                except Exception as e:
                    self.status_signal.emit(f'Error: {e}')
                    break

    def stop(self):
        self._run_flag = False
        self.wait()

class AudioTranslateWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.results = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel('AUDIO TO SIGN TRANSLATOR')
        header.setFont(QFont('Segoe UI', 18, QFont.Bold))
        header.setStyleSheet(f'color: {ACCENT};')
        layout.addWidget(header)

        self.status_lbl = QLabel('Choose an option to begin')
        self.status_lbl.setStyleSheet('color: #aaa; font-style: italic;')
        layout.addWidget(self.status_lbl)

        btn_lay = QHBoxLayout()
        self.start_btn = QPushButton('START MIC LISTENING')
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet(f'background: {ACCENT}; color: black; font-weight: bold; border-radius: 8px;')
        self.start_btn.clicked.connect(self.toggle_listening)
        btn_lay.addWidget(self.start_btn)

        self.upload_btn = QPushButton('UPLOAD AUDIO FILE')
        self.upload_btn.setFixedHeight(40)
        self.upload_btn.setStyleSheet(f'background: #333; color: white; font-weight: bold; border-radius: 8px;')
        self.upload_btn.clicked.connect(self.upload_audio)
        btn_lay.addWidget(self.upload_btn)
        
        self.exp_h_btn = QPushButton('Export Hybrid')
        self.exp_h_btn.setFixedHeight(40)
        self.exp_h_btn.setEnabled(False)
        self.exp_h_btn.clicked.connect(lambda: self.export_signs("hybrid"))
        btn_lay.addWidget(self.exp_h_btn)
        
        self.exp_a_btn = QPushButton('Export AI')
        self.exp_a_btn.setFixedHeight(40)
        self.exp_a_btn.setEnabled(False)
        self.exp_a_btn.clicked.connect(lambda: self.export_signs("ai"))
        btn_lay.addWidget(self.exp_a_btn)

        layout.addLayout(btn_lay)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet('background: transparent; border: none;')
        self.container = QWidget()
        self.flow_layout = QVBoxLayout(self.container)
        self.flow_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        self.worker = None

    def upload_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if path:
            self.clear_ui()
            self.worker = AudioWorker(mode="file", file_path=path)
            self.worker.heard_signal.connect(self.on_heard)
            self.worker.status_signal.connect(self.status_lbl.setText)
            self.worker.start()

    def toggle_listening(self):
        if self.worker and self.worker.isRunning() and self.worker.mode == "mic":
            self.worker.stop()
            self.worker = None
            self.start_btn.setText('START MIC LISTENING')
            self.start_btn.setStyleSheet(f'background: {ACCENT}; color: black; font-weight: bold; border-radius: 8px;')
            self.status_lbl.setText('Stopped.')
        else:
            if self.worker and self.worker.isRunning(): self.worker.stop()
            self.clear_ui()
            self.worker = AudioWorker(mode="mic")
            self.worker.heard_signal.connect(self.on_heard)
            self.worker.status_signal.connect(self.status_lbl.setText)
            self.worker.start()
            self.start_btn.setText('STOP MIC LISTENING')
            self.start_btn.setStyleSheet('background: #ff4444; color: white; font-weight: bold; border-radius: 8px;')

    def clear_ui(self):
        while self.flow_layout.count():
            item = self.flow_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.results = []
        self.exp_h_btn.setEnabled(False)
        self.exp_a_btn.setEnabled(False)

    def on_heard(self, text):
        self.clear_ui()
        self.results = match_gifs(text)
        for word, hybrid, ai in self.results:
            self.flow_layout.addWidget(WordCard(word, hybrid, ai))
        
        if self.results:
            self.exp_h_btn.setEnabled(True)
            self.exp_a_btn.setEnabled(True)

    def export_signs(self, mode):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", f"audio_signs_{mode}.mp4", "MP4 Video (*.mp4)")
        if not path: return
        items = [(w, h if mode=="hybrid" else a) for w, h, a in self.results]
        self.export_worker = ExportWorker(items, path)
        self.export_worker.progress_signal.connect(lambda p, s: self.progress.setValue(p))
        self.export_worker.done_signal.connect(lambda f: QMessageBox.information(self, "Done", f"Video saved to {f}"))
        self.export_worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.progress.setVisible(True)
        self.export_worker.start()
