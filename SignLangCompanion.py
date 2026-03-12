import sys
import os
import time
import subprocess
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTabWidget, QPushButton, QFrame, QProgressBar, QTextEdit,
    QLineEdit, QFileDialog, QMessageBox
)
import traceback

def exception_hook(exctype, value, tb):
    err_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(f'\n[CRITICAL ERROR] Application crashed!\n{err_msg}')
    with open('crash_log.txt', 'w', encoding='utf-8') as f:
        f.write(err_msg)
    QMessageBox.critical(None, 'Application Error', f'A fatal error occurred:\n{value}\n\nCheck crash_log.txt for details.')
    sys.exit(1)

sys.excepthook = exception_hook

import importlib
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    app_ui = importlib.import_module('3_app_ui')
    video_translate = importlib.import_module('4_video_translate')
    text_translate = importlib.import_module('5_text_to_sign')
    audio_translate = importlib.import_module('8_audio_translate')

    CommunicatorWidget = app_ui.CommunicatorWidget
    TranslatorWidget = video_translate.TranslatorWidget
    TextTranslateWidget = text_translate.TextTranslateWidget
    AudioTranslateWidget = audio_translate.AudioTranslateWidget
except Exception as e:
    print(f'[!] Error importing modules: {e}')
    traceback.print_exc()
    # Fallback to dummy widgets with required methods to prevent crash
    class CommunicatorWidget(QWidget):
        def pause(self): pass
        def resume(self): pass
    class TranslatorWidget(QWidget): pass
    class TextTranslateWidget(QWidget): pass
    class AudioTranslateWidget(QWidget):
        def toggle_listening(self): pass

BG_DARK = '#121212'
ACCENT  = '#00d7ff'

class TrainerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)

    def run(self):
        try:
            process = subprocess.Popen(
                [sys.executable, '2_train_model.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                self.log_signal.emit(line.strip())
                if 'Epoch' in line and '/' in line:
                    try:
                        parts = line.split()
                        epoch_part = [p for p in parts if '/' in p][0]
                        curr, total = map(int, epoch_part.split('/'))
                        self.progress_signal.emit(int((curr / total) * 100))
                    except: pass
            process.wait()
            self.finished_signal.emit(process.returncode == 0)
        except Exception as e:
            self.log_signal.emit(f'Error starting trainer: {e}')
            self.finished_signal.emit(False)

class SignLangCompanion(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sign Language Companion - Unified Workspace')
        self.resize(1280, 850)
        self.setStyleSheet(f"background-color: {BG_DARK}; color: white; font-family: 'Segoe UI', sans-serif;")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f'background-color: #1e1e1e; border-bottom: 2px solid {ACCENT};')
        h_layout = QHBoxLayout(header)
        logo = QLabel('SIGN LANGUAGE COMPANION')
        logo.setFont(QFont('Segoe UI', 18, QFont.Bold))
        logo.setStyleSheet(f'color: {ACCENT}; padding-left: 20px;')
        h_layout.addWidget(logo)
        h_layout.addStretch()
        layout.addWidget(header)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {BG_DARK}; }}
            QTabBar::tab {{
                background: #252525; color: #888; padding: 12px 25px;
                border-top-left-radius: 8px; border-top-right-radius: 8px;
                margin-right: 4px; font-weight: bold;
            }}
            QTabBar::tab:selected {{ background: #333; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
            QTabBar::tab:hover {{ background: #2a2a2a; color: white; }}
        """)
        self.comm_tab = CommunicatorWidget()
        self.tabs.addTab(self.comm_tab, 'COMMUNICATOR')
        self.trans_tab = TranslatorWidget()
        self.tabs.addTab(self.trans_tab, 'VIDEO TRANSLATE')
        self.text_tab = TextTranslateWidget()
        self.tabs.addTab(self.text_tab, 'TEXT TRANSLATE')
        self.audio_tab = AudioTranslateWidget()
        self.tabs.addTab(self.audio_tab, 'AUDIO TRANSLATE')
        self.data_tab = self.create_data_tab()
        self.tabs.addTab(self.data_tab, 'DATASET BUILDER')
        self.train_tab = self.create_train_tab()
        self.tabs.addTab(self.train_tab, 'TRAINER')
        self.tabs.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabs)

    def on_tab_changed(self, index):
        if index == 0:
            if hasattr(self.comm_tab, 'resume'): self.comm_tab.resume()
        else:
            if hasattr(self.comm_tab, 'pause'): self.comm_tab.pause()
            
        if index != 1 and hasattr(self.trans_tab, 'player_thread'):
            try: self.trans_tab.player_thread.pause()
            except: pass
        if index != 3 and hasattr(self.audio_tab, 'toggle_listening'):
            if hasattr(self.audio_tab, 'worker') and self.audio_tab.worker:
                if self.audio_tab.worker.isRunning(): self.audio_tab.toggle_listening()

    def create_data_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(50, 50, 50, 50)
        lbl = QLabel('Manage Your Dataset')
        lbl.setFont(QFont('Segoe UI', 24, QFont.Bold))
        layout.addWidget(lbl)
        desc = QLabel('Add new signs by recording yourself or downloading from the MS-ASL cloud.')
        desc.setStyleSheet('color: #aaa; font-size: 14px;')
        layout.addWidget(desc)
        layout.addSpacing(30)
        grid = QHBoxLayout()
        rec_card = QFrame()
        rec_card.setStyleSheet('background: #1e1e1e; border-radius: 12px; border: 1px solid #333;')
        rec_lay = QVBoxLayout(rec_card)
        rec_lay.setContentsMargins(30, 30, 30, 30)
        rec_icon = QLabel('CAM')
        rec_icon.setFont(QFont('Segoe UI', 32))
        rec_icon.setAlignment(Qt.AlignCenter)
        rec_lay.addWidget(rec_icon)
        rec_title = QLabel('Manual Record')
        rec_title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        rec_title.setAlignment(Qt.AlignCenter)
        rec_lay.addWidget(rec_title)
        rec_btn = QPushButton('Start Recording AI')
        rec_btn.setStyleSheet(f"background: {ACCENT}; color: black; border-radius: 6px; padding: 12px; font-weight: bold;")
        rec_btn.clicked.connect(self.run_recorder)
        rec_lay.addWidget(rec_btn)
        grid.addWidget(rec_card)
        dl_card = QFrame()
        dl_card.setStyleSheet('background: #1e1e1e; border-radius: 12px; border: 1px solid #333;')
        dl_lay = QVBoxLayout(dl_card)
        dl_lay.setContentsMargins(30, 30, 30, 30)
        dl_icon = QLabel('CLOUD')
        dl_icon.setFont(QFont('Segoe UI', 32))
        dl_icon.setAlignment(Qt.AlignCenter)
        dl_lay.addWidget(dl_icon)
        dl_title = QLabel('Cloud Download')
        dl_title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        dl_title.setAlignment(Qt.AlignCenter)
        dl_lay.addWidget(dl_title)
        self.dl_input = QLineEdit()
        self.dl_input.setPlaceholderText('Enter word to download')
        self.dl_input.setStyleSheet('background: #2a2a2a; border: none; padding: 10px; border-radius: 4px;')
        dl_lay.addWidget(self.dl_input)
        dl_btn = QPushButton('Fetch from MS-ASL')
        dl_btn.setStyleSheet(f"background: #00882b; color: white; border-radius: 6px; padding: 12px; font-weight: bold;")
        dl_btn.clicked.connect(self.run_downloader)
        dl_lay.addWidget(dl_btn)
        grid.addWidget(dl_card)
        layout.addLayout(grid)
        layout.addStretch()
        return widget

    def create_train_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 40, 40, 40)
        header = QLabel('Neural Network Trainer')
        header.setFont(QFont('Segoe UI', 20, QFont.Bold))
        layout.addWidget(header)
        self.train_status = QLabel('Ready to optimize model.')
        self.train_status.setStyleSheet('color: #aaa;')
        layout.addWidget(self.train_status)
        self.train_btn = QPushButton('START TRAINING')
        self.train_btn.setFixedHeight(50)
        self.train_btn.setStyleSheet(f"background: {ACCENT}; color: black; font-weight: bold; border-radius: 8px;")      
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)
        self.train_progress = QProgressBar()
        self.train_progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {ACCENT}; }}")
        layout.addWidget(self.train_progress)
        self.train_logs = QTextEdit()
        self.train_logs.setReadOnly(True)
        self.train_logs.setStyleSheet('background: #000; color: #00ff00; font-family: Consolas, monospace;')      
        layout.addWidget(self.train_logs)
        return widget

    def run_recorder(self):
        subprocess.Popen(['start', 'cmd', '/c', sys.executable, '1_collect_data.py'], shell=True)

    def run_downloader(self):
        word = self.dl_input.text().strip()
        if word: subprocess.Popen(['start', 'cmd', '/c', sys.executable, '7_msasl_downloader.py', '--words', word], shell=True)

    def start_training(self):
        self.train_btn.setEnabled(False)
        self.trainer_thread = TrainerThread()
        self.trainer_thread.log_signal.connect(self.train_logs.append)
        self.trainer_thread.progress_signal.connect(self.train_progress.setValue)
        self.trainer_thread.finished_signal.connect(self.on_training_finished)
        self.trainer_thread.start()

    def on_training_finished(self, s):
        self.train_btn.setEnabled(True)
        self.train_status.setText('Complete!' if s else 'Failed.')

    def closeEvent(self, event):
        try: self.comm_tab.closeEvent(event)
        except: pass
        try: self.trans_tab.closeEvent(event)
        except: pass
        if hasattr(self.audio_tab, 'worker') and self.audio_tab.worker:
            self.audio_tab.worker.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignLangCompanion()
    window.show()
    sys.exit(app.exec_())
