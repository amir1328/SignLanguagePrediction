import sys
import os

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QMovie
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QSizePolicy,
    QTextEdit
)

GIF_DIR = "sign_gifs"

try:
    from asl_gloss import english_to_asl, asl_gloss_string
except ImportError:
    # Fallback: no conversion, plain split
    def english_to_asl(text): return text.lower().split()
    def asl_gloss_string(text): return text.upper()

ACCENT   = "#00d7ff"
BG_DARK  = "#1a1a1a"
BG_PANEL = "#2b2b2b"
BG_CARD  = "#222222"


# =============================================================================
# Word to GIF Matcher
# =============================================================================
def match_gifs(text):
    """
    Convert English text to ASL gloss, then match each token to a sign GIF.
    Returns list of (word, path_or_None).
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
                    results.append((phrase, gif_path))
                    i += length; matched = True; break
                elif os.path.exists(mp4_path):
                    results.append((phrase, mp4_path))
                    i += length; matched = True; break
            if matched:
                break
        if not matched:
            results.append((asl_tokens[i], None))
            i += 1
    return results


# =============================================================================
# Word Card Widget
# =============================================================================
class WordCard(QFrame):
    def __init__(self, word, gif_path=None):
        super().__init__()
        has_gif = gif_path is not None
        self.setFixedSize(160, 200)
        border = ACCENT if has_gif else "#3a3a3a"
        bg     = "#1a2a1a" if has_gif else BG_CARD
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border-radius: 8px; border: 2px solid {border}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        gif_lbl = QLabel()
        gif_lbl.setFixedSize(144, 108)
        gif_lbl.setAlignment(Qt.AlignCenter)
        gif_lbl.setStyleSheet("border: none; background: transparent;")
        if has_gif:
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(QSize(144, 108))
            gif_lbl.setMovie(self.movie)
            self.movie.start()
        else:
            self.movie = None
            gif_lbl.setStyleSheet("color: #555555; font-size: 20px; border: none;")
            gif_lbl.setText("—")
        layout.addWidget(gif_lbl)

        word_lbl = QLabel(word)
        word_lbl.setAlignment(Qt.AlignCenter)
        word_lbl.setWordWrap(True)
        word_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        word_lbl.setStyleSheet("color: white; border: none;")
        layout.addWidget(word_lbl)

        status = QLabel("matched" if has_gif else "no sign")
        status.setAlignment(Qt.AlignCenter)
        status.setFont(QFont("Segoe UI", 8))
        status.setStyleSheet(f"color: {ACCENT if has_gif else '#666666'}; border: none;")
        layout.addWidget(status)


# =============================================================================
# Main Window
# =============================================================================
class TextToSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text to Sign Language")
        self.resize(1100, 650)
        self.setStyleSheet(
            f"background-color: {BG_DARK}; color: white; font-family: Segoe UI, sans-serif;"
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # Header
        header = QLabel("Text to Sign Language")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        root.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {ACCENT};")
        root.addWidget(sep)

        # Input area
        input_panel = QWidget()
        input_panel.setStyleSheet(
            f"background-color: {BG_PANEL}; border-radius: 12px;"
        )
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(14, 14, 14, 14)
        input_layout.setSpacing(10)

        input_hint = QLabel("Type or paste text below:")
        input_hint.setFont(QFont("Segoe UI", 11))
        input_hint.setStyleSheet("color: #aaaaaa;")
        input_layout.addWidget(input_hint)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("e.g.  hello what's up")
        self.text_input.setFixedHeight(80)
        self.text_input.setFont(QFont("Segoe UI", 13))
        self.text_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: #111111;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 8px;
            }}
            QTextEdit:focus {{
                border: 1px solid {ACCENT};
            }}
        """)
        input_layout.addWidget(self.text_input)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(32)
        clear_btn.setFixedWidth(80)
        clear_btn.setStyleSheet(self._btn("#3a3a3a", "#4a4a4a"))
        clear_btn.clicked.connect(self.clear_all)
        btn_row.addWidget(clear_btn)

        translate_btn = QPushButton("Translate")
        translate_btn.setFixedHeight(32)
        translate_btn.setFixedWidth(100)
        translate_btn.setStyleSheet(self._btn("#00882b", "#006622"))
        translate_btn.clicked.connect(self.translate)
        btn_row.addWidget(translate_btn)

        input_layout.addLayout(btn_row)
        root.addWidget(input_panel)

        # Status line
        self.status_lbl = QLabel("Enter text above and click Translate.")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        root.addWidget(self.status_lbl)

        # Sign cards panel — matches right panel from 3_app_ui.py
        right_panel = QWidget()
        right_panel.setStyleSheet(
            f"background-color: {BG_PANEL}; border-radius: 15px;"
        )
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(12, 12, 12, 12)
        rp_layout.setSpacing(8)

        cards_header = QLabel("Sign Translations")
        cards_header.setFont(QFont("Segoe UI", 13, QFont.Bold))
        cards_header.setAlignment(Qt.AlignCenter)
        rp_layout.addWidget(cards_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_flow = QHBoxLayout(self.cards_container)
        self.cards_flow.setAlignment(Qt.AlignLeft)
        self.cards_flow.setSpacing(10)
        self.cards_flow.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self.cards_container)
        rp_layout.addWidget(scroll)

        root.addWidget(right_panel, stretch=1)

        # Allow pressing Ctrl+Enter to translate
        self.text_input.installEventFilter(self)

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent
        if obj is self.text_input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() == Qt.ControlModifier:
                self.translate()
                return True
        return super().eventFilter(obj, event)

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

    def translate(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_lbl.setText("Please enter some text first.")
            return

        self._clear_cards()
        gloss = asl_gloss_string(text)
        matched = match_gifs(text)

        for word, gif_path in matched:
            self.cards_flow.addWidget(WordCard(word, gif_path))

        found = sum(1 for _, p in matched if p)
        total = len(matched)
        self.status_lbl.setText(
            f"ASL Gloss: {gloss}  |  {found}/{total} signs matched."
            + ("" if found == total else f"  ({total - found} words have no recorded sign yet.)")
        )

    def clear_all(self):
        self.text_input.clear()
        self._clear_cards()
        self.status_lbl.setText("Enter text above and click Translate.")

    def _clear_cards(self):
        while self.cards_flow.count():
            item = self.cards_flow.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TextToSignApp()
    win.show()
    sys.exit(app.exec_())
