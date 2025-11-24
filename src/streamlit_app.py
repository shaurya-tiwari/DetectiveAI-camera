import streamlit as st
import av
import cv2
import time
import os
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Imports for your AI Pipeline ---
from src.detection import Detector
from src.tracking import Tracker
from src.rules import RuleEngine
from src.visualize import draw_tracks, draw_detections, draw_alerts


# =======================
# STREAMLIT SETUP
# =======================
st.set_page_config(page_title="SentinelAI Dashboard", layout="wide")

st.title("SentinelAI: Intelligent Surveillance System (Smooth Version)")
st.markdown("Optimized real-time surveillance with YOLO + WebRTC + Tracking + Rule Engine.")


# =======================
# CONFIG
# =======================
MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.40
CROWD_THRESHOLD = 20
BAG_STATIONARY_SECONDS = 5
WEAPON_PERSIST_FRAMES = 5


# =======================
# VIDEO PROCESSOR CLASS
# =======================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = Detector(model_path=MODEL_PATH)
        self.detector.class_remap = {}  # ensure no remapping

        self.tracker = Tracker()
        self.rules = RuleEngine(
            crowd_threshold=CROWD_THRESHOLD,
            bag_stationary_seconds=BAG_STATIONARY_SECONDS,
            weapon_persist_frames=WEAPON_PERSIST_FRAMES
        )

        self.frame_idx = 0
        self.alert_history = []

    def recv(self, frame):
        """
        WebRTC automatically sends frames here
        """

        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1

        # Run detection (every frame - GPU handles it)
        detections = self.detector.detect(img, conf_threshold=CONF_THRESHOLD)

        # Tracking
        tracks = self.tracker.update(detections, img)

        # Apply rules
        alerts = self.rules.process(
            tracks,
            frame_idx=self.frame_idx,
            frame_timestamp=time.time()
        )

        # Update alert history
        if alerts:
            for alert in alerts:
                timestamp = time.strftime("%H:%M:%S")
                self.alert_history.insert(
                    0, f"{timestamp} - [{alert['type']}] {alert['message']}"
                )
            self.alert_history = self.alert_history[:20]  # keep latest 20

        # Draw overlays
        img = draw_tracks(img, tracks)
        img = draw_alerts(img, alerts)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =======================
# SIDEBAR UI
# =======================
st.sidebar.header("Video Source")

source = st.sidebar.radio("Choose Source", ["Webcam", "Upload Video"])

uploaded_video_path = None

if source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        uploaded_video_path = tfile.name


# =======================
# LAYOUT
# =======================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video")

with col2:
    st.subheader("Real-Time Alerts")
    alert_panel = st.empty()


# =======================
# RUN WEBRTC STREAMER
# =======================
ctx = webrtc_streamer(
    key="SentinelAI",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# =======================
# UPDATE ALERT PANEL
# =======================
if ctx.video_processor:
    alert_history = ctx.video_processor.alert_history
    alert_panel.markdown(
        "<br>".join(alert_history),
        unsafe_allow_html=True
    )
