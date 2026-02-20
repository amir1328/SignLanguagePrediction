"""
Sign Language Recognition — Streamlit App
==========================================
Uses cv2.VideoCapture(0) directly → always the system default (laptop) camera.
Run:  streamlit run app.py
"""

import os, csv, time, threading, queue, pickle
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pythoncom, pyttsx3

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_FILE       = "dataset.csv"
MODEL_FILE         = "sign_model.pkl"
FRAMES_PER_SIGN    = 300
PREDICTION_HISTORY = 8
TTS_COOLDOWN       = 1.5
HEADER = [f"{ax}{i}" for i in range(21) for ax in ("x","y","z")] + ["label"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🖐",
    layout="wide",
)
st.markdown("""
<style>
  .big-sign { font-size:3rem; font-weight:700; color:#00c48c; text-align:center; margin:8px 0; }
  .heard-text { font-size:1.4rem; color:#f0c040; text-align:center; }
  .stTabs [data-baseweb="tab"] { font-size:1.05rem; }
  .status-pill { display:inline-block; padding:4px 14px; border-radius:20px;
                 font-weight:600; font-size:.9rem; }
  .pill-on  { background:#00c48c22; color:#00c48c; border:1px solid #00c48c; }
  .pill-off { background:#88888822; color:#888;    border:1px solid #555; }
</style>
""", unsafe_allow_html=True)

st.title("🖐 Sign Language Recognition")

# ── Auto-refresh (drives live display at ~10 fps) ─────────────────────────────
st_autorefresh(interval=75, key="global_refresh")

# ── TTS singleton ─────────────────────────────────────────────────────────────
@st.cache_resource
def _start_tts():
    q: queue.Queue = queue.Queue()
    def _worker():
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        while True:
            text = q.get()
            if text is None:
                break
            print(f"[Streamlit TTS] Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
        pythoncom.CoUninitialize()
    threading.Thread(target=_worker, daemon=True).start()
    return q

_tts_q = _start_tts()

def speak(text: str):
    while not _tts_q.empty():
        try: _tts_q.get_nowait()
        except queue.Empty: pass
    _tts_q.put(text)

# ── Camera Manager singleton ───────────────────────────────────────────────────
class _CameraManager:
    """
    Background thread: opens cv2.VideoCapture(0) (always the default camera),
    runs MediaPipe, and exposes the latest annotated frame + predicted sign.
    """
    def __init__(self):
        self._lock          = threading.Lock()
        self._frame         = None       # latest BGR numpy frame
        self._sign          = ""
        self._mode          = "idle"     # "idle" | "collect" | "live"
        self._label         = ""
        self._count         = 0
        self._target        = FRAMES_PER_SIGN
        self._running       = False
        self._thread        = None

    # ── public controls ───────────────────────────────────────────────────────
    def start_idle(self):
        self._set_mode("idle")
        self._ensure_thread()

    def start_collect(self, label: str, target: int):
        with self._lock:
            self._label  = label
            self._target = target
            self._count  = 0
            self._mode   = "collect"
        self._ensure_thread()

    def start_live(self):
        self._set_mode("live")
        self._ensure_thread()

    def stop(self):
        with self._lock:
            self._running = False
            self._mode    = "idle"

    # ── public reads ─────────────────────────────────────────────────────────
    @property
    def frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def sign(self):
        with self._lock:
            return self._sign

    @property
    def collect_count(self):
        with self._lock:
            return self._count

    @property
    def collect_target(self):
        with self._lock:
            return self._target

    @property
    def mode(self):
        with self._lock:
            return self._mode

    # ── internals ─────────────────────────────────────────────────────────────
    def _set_mode(self, mode: str):
        with self._lock:
            self._mode = mode

    def _ensure_thread(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        mp_draw  = mp.solutions.drawing_utils
        mp_conns = mp.solutions.hands.HAND_CONNECTIONS

        # Load model if it exists
        model = None
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)

        backends = [
            ("DirectShow (DSHOW)", cv2.CAP_DSHOW),
            ("Media Foundation (MSMF)", cv2.CAP_MSMF),
            ("Default (ANY)", cv2.CAP_ANY)
        ]
        indices = [1, 0, 2]
        
        cap = None
        for index in indices:
            for name, backend in backends:
                print(f"[*] Streamlit app trying Camera Index {index} with {name}...")
                temp_cap = cv2.VideoCapture(index, backend)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret and frame is not None:
                        print(f"[✔] SUCCESS: Camera {index} opened with {name}!")
                        cap = temp_cap
                        break
                    else:
                        temp_cap.release()
            if cap is not None:
                break

        if cap is None or not cap.isOpened():
            with self._lock:
                self._running = False
            return

        history       = []
        last_spoken   = ""
        last_at       = 0.0
        csv_file      = None
        csv_writer    = None

        while True:
            with self._lock:
                if not self._running:
                    break
                mode   = self._mode
                label  = self._label
                target = self._target
                count  = self._count

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            sign = ""
            if result.multi_hand_landmarks:
                for hl in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_conns)
                    feats = [v for lm in hl.landmark for v in (lm.x, lm.y, lm.z)]

                    # ── collect mode ─────────────────────────────────────────
                    if mode == "collect" and count < target:
                        if csv_file is None:
                            csv_file   = open(DATASET_FILE, "a", newline="")
                            csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(feats + [label])
                        with self._lock:
                            self._count += 1
                            count = self._count
                        if count >= target:
                            if csv_file:
                                csv_file.close()
                                csv_file = csv_writer = None
                            with self._lock:
                                self._mode = "idle"
                                mode = "idle"

                    # ── live mode ─────────────────────────────────────────────
                    elif mode == "live" and model:
                        pred = model.predict([feats])[0]
                        history.append(pred)
                        if len(history) > PREDICTION_HISTORY:
                            history.pop(0)
                        sign = max(set(history), key=history.count)
            else:
                if mode == "live":
                    history.append("")
                    if len(history) > PREDICTION_HISTORY:
                        history.pop(0)
                    if history:
                        sign = max(set(history), key=history.count)

            with self._lock:
                self._sign = sign

            # TTS
            now = time.time()
            if mode == "live":
                if sign and sign != "" and sign != last_spoken and (now - last_at) >= TTS_COOLDOWN:
                    print(f"[Streamlit TTS Trigger] Queuing speech: {sign}")
                    speak(sign)
                    last_spoken = sign
                    last_at     = now
                elif sign == "":
                    # Reset so we can speak the same sign again after hand is removed
                    last_spoken = ""

            # ── overlay ───────────────────────────────────────────────────────
            if mode == "collect":
                pct   = min(count / max(target, 1), 1.0)
                bar_w = int(w * pct)
                cv2.rectangle(frame, (0, h-10), (bar_w, h), (0, 200, 90), -1)
                cv2.putText(frame, f"Recording '{label}': {count}/{target}",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
            elif mode == "live":
                display = sign if sign else "No hand detected"
                cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 0), -1)
                cv2.putText(frame, f"Sign: {display}",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2)
            else:
                cv2.putText(frame, "Camera ready — select a mode below",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)

            with self._lock:
                self._frame = frame

        cap.release()
        if csv_file:
            csv_file.close()
        with self._lock:
            self._running = False

@st.cache_resource
def get_camera():
    m = _CameraManager()
    m.start_idle()   # warm up camera immediately
    return m

cam = get_camera()

def _frame_to_jpeg(frame) -> bytes:
    """Encode a BGR numpy frame to JPEG bytes for stable st.image display."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()

import base64
def show_frame(frame):
    """Display a BGR frame via inline base64 HTML — no Streamlit media cache."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode()
    st.markdown(
        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px;">',
        unsafe_allow_html=True,
    )

# ── Ensure dataset header ──────────────────────────────────────────────────────
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷 Collect Data", "🧠 Train Model", "🖐 Live Recognize"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Collect Data
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Step 1 — Collect Training Data")

    ctrl_col, cam_col = st.columns([1, 2])

    with ctrl_col:
        label_input   = st.text_input("Sign label", placeholder="e.g. Hello")
        target_frames = st.slider("Frames to capture", 100, 1000, FRAMES_PER_SIGN, 50)

        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ Start", type="primary",
                              disabled=not label_input.strip(), key="collect_start")
        stop_btn  = c2.button("⏹ Stop", key="collect_stop")

        if start_btn and label_input.strip():
            cam.start_collect(label_input.strip(), target_frames)
        if stop_btn:
            cam.start_idle()

        # Progress
        cnt = cam.collect_count
        tgt = cam.collect_target
        if cam.mode == "collect" and tgt > 0:
            st.progress(min(cnt / tgt, 1.0), text=f"Collected {cnt} / {tgt} frames")
        elif cnt >= tgt and tgt > 0:
            st.success(f"✅ Done! {cnt} frames collected.")

        st.markdown("---")
        st.markdown("**Dataset summary**")
        if os.path.exists(DATASET_FILE):
            try:
                df_prev = pd.read_csv(DATASET_FILE)
                if not df_prev.empty:
                    st.dataframe(
                        df_prev["label"].value_counts().rename("frames"),
                        width="stretch",
                    )
                else:
                    st.info("Dataset is empty — start collecting!")
            except Exception:
                st.info("No data yet.")
        else:
            st.info("No dataset file yet.")

    with cam_col:
        frame = cam.frame
        if frame is not None:
            show_frame(frame)
        else:
            st.info("⏳ Camera starting…")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Train Model
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Step 2 — Train the Model")

    if not os.path.exists(DATASET_FILE):
        st.warning("No dataset found. Collect some data first (Tab 1).")
    else:
        try:
            df_train = pd.read_csv(DATASET_FILE)
        except Exception:
            df_train = pd.DataFrame()

        if df_train.empty:
            st.warning("Dataset is empty. Go collect data first.")
        else:
            labels = sorted(df_train["label"].unique())
            m1, m2, m3 = st.columns(3)
            m1.metric("Total frames",   len(df_train))
            m2.metric("Unique signs",   len(labels))
            m3.metric("Features/frame", 63)
            st.markdown(f"**Signs:** {', '.join(labels)}")
            st.markdown("---")

            n_trees   = st.slider("Number of trees",  50, 500, 200, 50)
            test_pct  = st.slider("Test split %",      10, 40, 20,  5)

            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training RandomForestClassifier…"):
                    X = df_train.drop(columns=["label"]).values
                    y = df_train["label"].values
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_pct/100, random_state=42, stratify=y
                    )
                    clf = RandomForestClassifier(
                        n_estimators=n_trees, random_state=42, n_jobs=-1
                    )
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    acc    = accuracy_score(y_te, y_pred)
                    with open(MODEL_FILE, "wb") as f:
                        pickle.dump(clf, f)

                st.success(f"✅ Model saved — Test accuracy: **{acc*100:.1f}%**")
                report_df = pd.DataFrame(
                    classification_report(y_te, y_pred, output_dict=True)
                ).T.round(2)
                st.dataframe(report_df, width="stretch")

            if os.path.exists(MODEL_FILE):
                st.info(f"💾 Trained model found: `{MODEL_FILE}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Recognize
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Step 3 — Live Recognition")

    if not os.path.exists(MODEL_FILE):
        st.error("No model found. Train one first in Tab 2.")
    else:
        ctrl2, cam2 = st.columns([1, 2])

        with ctrl2:
            lstart = st.button("▶ Start Recognition", type="primary", key="live_start")
            lstop  = st.button("⏹ Stop",                               key="live_stop")

            if lstart:
                cam.start_live()
            if lstop:
                cam.start_idle()

            # Mode pill
            mode = cam.mode
            if mode == "live":
                st.markdown('<span class="status-pill pill-on">🟢 Live</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-pill pill-off">⚫ Idle</span>',
                            unsafe_allow_html=True)

            # Big predicted sign
            sign = cam.sign
            if sign:
                st.markdown(f'<div class="big-sign">🖐 {sign}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="text-align:center;font-size:1.1rem;color:#666;margin-top:16px">'
                    'No hand detected</div>',
                    unsafe_allow_html=True,
                )

        with cam2:
            frame2 = cam.frame
            if frame2 is not None:
                show_frame(frame2)
            else:
                st.info("⏳ Camera starting…")
