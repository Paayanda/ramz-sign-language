from collections import deque
from pathlib import Path
from statistics import mode
import threading

from arabic_reshaper import reshape
from bidi.algorithm import get_display
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

st.set_page_config(
    page_title="تشخیص زنده و کلمه‌سازی الفبای اشاره - RAMZ",
    page_icon="✋",
    layout="wide",
)


def load_css(css_filename="style.css"):
    css_path = Path(__file__).parent / css_filename
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css("style.css")

fa_labels = {
    "alef": "ا",
    "be": "ب",
    "che": "چ",
    "dal": "د",
    "heh": "ح",
    "khe": "خ",
    "pe": "پ",
    "se": "ث",
    "te": "ت",
    "zal": "ذ",
}

FEATURE_COLS = [f"{a}{i}" for i in range(21) for a in "xyz"]

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}


@st.cache_resource
def load_models():
    clf = joblib.load("ramz_model.joblib")
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return clf, detector


def to_features(hand_landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks])
    pts = pts - pts[0]
    size = np.linalg.norm(pts[9])
    if size == 0:
        return None
    return (pts / size).flatten()


def draw_landmarks_on_bgr(frame_bgr, hand_landmarks):
    h, w, _ = frame_bgr.shape
    points_2d = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points_2d.append((px, py))
        cv2.circle(frame_bgr, (px, py), 4, (0, 255, 255), -1)

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    ]
    for i, j in connections:
        cv2.line(frame_bgr, points_2d[i], points_2d[j], (255, 150, 0), 2)
    return frame_bgr


def put_persian_text(
    frame_bgr, text, org=(30, 30), font_size=30, color=(0, 255, 0)
):
    reshaped = reshape(text)
    bidi_text = get_display(reshaped)
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(org, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self, clf, detector, threshold=0.6):
        self.clf = clf
        self.detector = detector
        self.threshold = threshold
        self.prediction_buffer = deque(maxlen=7)
        self.formed_text = ""
        self.last_added_char = ""
        self.cooldown_counter = 0
        self.lock = threading.Lock()

    def add_space(self):
        with self.lock:
            self.formed_text += " "
            self.last_added_char = ""

    def backspace(self):
        with self.lock:
            self.formed_text = self.formed_text[:-1]
            self.last_added_char = ""

    def clear_text(self):
        with self.lock:
            self.formed_text = ""
            self.last_added_char = ""
            self.prediction_buffer.clear()
            self.cooldown_counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        current_char_display = "ناشناس"

        with self.lock:
            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                img = draw_landmarks_on_bgr(img, hand_lms)
                feats = to_features(hand_lms)

                if feats is not None:
                    feats_df = pd.DataFrame([feats], columns=FEATURE_COLS)
                    proba = self.clf.predict_proba(feats_df)[0]
                    pred_key = self.clf.classes_[proba.argmax()]
                    conf = float(proba.max())

                    if conf >= self.threshold:
                        fa_char = fa_labels.get(pred_key, pred_key)
                        self.prediction_buffer.append(fa_char)

                        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                            try:
                                stable_char = mode(self.prediction_buffer)
                            except Exception:
                                stable_char = fa_char
                        else:
                            stable_char = fa_char

                        current_char_display = f"حرف: {stable_char} ({conf:.2f})"

                        if self.cooldown_counter == 0:
                            if stable_char != self.last_added_char:
                                self.formed_text += stable_char
                                self.last_added_char = stable_char
                                self.cooldown_counter = 15
                    else:
                        current_char_display = "نامعلوم"
            else:
                self.prediction_buffer.clear()
                self.last_added_char = ""

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            text_to_show = self.formed_text

        img = put_persian_text(
            img,
            current_char_display,
            org=(30, 30),
            font_size=30,
            color=(0, 255, 0),
        )

        bottom_y = max(30, h - 75)
        box_text = f"متن: {text_to_show}"
        img = put_persian_text(
            img,
            box_text,
            org=(30, bottom_y),
            font_size=35,
            color=(255, 255, 0),
        )

        return frame.from_ndarray(img, format="bgr24")


st.markdown(
    """
    <div class="ramz-header">
        <h1>✋ تشخیص زنده و کلمه‌سازی الفبای اشاره - RAMZ</h1>
        <p>دست خود را جلوی دوربین بگیرید تا حروف تشخیص داده شوند و در پایین تصویر کلمه بسازند.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

clf, detector = load_models()
threshold = st.slider(
    "آستانه اطمینان (Confidence Threshold)", 0.1, 0.95, 0.6, 0.05
)

ctx = webrtc_streamer(
    key="ramz-detect-word-builder",
    video_processor_factory=lambda: SignLanguageProcessor(
        clf, detector, threshold
    ),
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("␣ فاصله (Space)", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.add_space()

with col2:
    if st.button("⌫ حذف آخرین حرف", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.backspace()

with col3:
    if st.button("🗑 پاک کردن کامل", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.clear_text()
