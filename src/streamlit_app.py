import streamlit as st
import cv2
import tempfile
import time
import os
import sys
import logging
import warnings

# Silence terminal noise
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(__file__))

from detection import Detector
from tracking import Tracker
from rules import RuleEngine
from visualize import draw_tracks, draw_alerts


@st.cache_resource
def load_model(model_path):
    return Detector(model_path=model_path)


# --- Page Config ---
st.set_page_config(page_title="SentinelAI", layout="wide", page_icon="🎯")

# --- Custom CSS: Clean white modern light UI ---
st.markdown("""
<style>
/* Import clean font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Global white background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit decoration */
[data-testid="stDecoration"], #MainMenu, footer { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid #e9ecef;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

/* Top header bar */
.sentinel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 24px 0 8px 0;
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 24px;
}
.sentinel-dot {
    width: 10px; height: 10px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.sentinel-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin: 0;
}
.sentinel-sub {
    font-size: 13px;
    color: #9ca3af;
    margin: 0;
}

/* Section labels */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #9ca3af;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Alert card */
.alert-card {
    background: #fff7ed;
    border-left: 3px solid #f97316;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #1f2937;
}
.alert-card-time {
    font-size: 11px;
    color: #9ca3af;
    margin-bottom: 2px;
}
.alert-card-text { font-weight: 500; color: #111827; }

/* Stat bar */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #f3f4f6;
    font-size: 13px;
}
.stat-label { color: #6b7280; }
.stat-value { font-weight: 500; color: #111827; }

/* Status badge */
.badge-running {
    display: inline-block;
    background: #dcfce7;
    color: #16a34a;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}
.badge-idle {
    display: inline-block;
    background: #f3f4f6;
    color: #6b7280;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}

/* Video container */
.video-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
}

/* Sidebar section header */
.sidebar-section {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 16px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)


# --- Config ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.onnx")
CONF_THRESHOLD = 0.20
CROWD_THRESHOLD = 20
BAG_STATIONARY_SECONDS = 5
WEAPON_PERSIST_FRAMES = 3

# --- Header ---
st.markdown("""
<div class="sentinel-header">
    <div class="sentinel-dot"></div>
    <div>
        <p class="sentinel-title">SentinelAI</p>
        <p class="sentinel-sub">Intelligent Surveillance — Weapon · Crowd · Bag Detection</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown('<div class="sidebar-section">Video Source</div>', unsafe_allow_html=True)
source_type = st.sidebar.radio("", ["Sample Video", "Upload Video", "Webcam"], label_visibility="collapsed")

video_path = None
if source_type == "Sample Video":
    video_folder = os.path.join(os.path.dirname(__file__), "..", "videos")
    if os.path.exists(video_folder):
        files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        if files:
            selected_file = st.sidebar.selectbox("", files, label_visibility="collapsed")
            video_path = os.path.join(video_folder, selected_file)
        else:
            st.sidebar.caption("No videos found in videos/ folder.")
    else:
        st.sidebar.caption("videos/ folder not found.")
elif source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif source_type == "Webcam":
    video_path = 0

st.sidebar.markdown('<div class="sidebar-section" style="margin-top:20px;">Controls</div>', unsafe_allow_html=True)

if "running" not in st.session_state:
    st.session_state.running = False

if st.sidebar.button("▶  Start", use_container_width=True, type="primary"):
    st.session_state.running = True
if st.sidebar.button("⏹  Stop", use_container_width=True):
    st.session_state.running = False

st.sidebar.divider()
st.sidebar.caption("Model: `best.onnx` · Classes: pistol, knife")
st.sidebar.caption("Threshold: 0.20 · Persist: 3 frames")

# --- Main Layout ---
col_video, col_panel = st.columns([3, 1], gap="large")

with col_video:
    st.markdown('<div class="section-label">Live Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

with col_panel:
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()

    st.markdown('<div class="section-label" style="margin-top:20px;">Alert Log</div>', unsafe_allow_html=True)
    alert_placeholder = st.empty()

# Idle state display
if not st.session_state.running:
    with col_video:
        st.markdown("""
        <div style="height:340px; background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px;
                    display:flex; align-items:center; justify-content:center; flex-direction:column; gap:8px;">
            <span style="font-size:32px">📹</span>
            <span style="font-size:14px; color:#9ca3af;">Select a source and press Start</span>
        </div>
        """, unsafe_allow_html=True)
    with col_panel:
        status_placeholder.markdown("""
        <div class="stat-row"><span class="stat-label">Status</span><span class="badge-idle">Idle</span></div>
        <div class="stat-row"><span class="stat-label">Frame</span><span class="stat-value">—</span></div>
        <div class="stat-row"><span class="stat-label">FPS</span><span class="stat-value">—</span></div>
        <div class="stat-row"><span class="stat-label">Tracks</span><span class="stat-value">—</span></div>
        """, unsafe_allow_html=True)
        alert_placeholder.markdown('<span style="font-size:13px; color:#9ca3af;">No alerts yet.</span>', unsafe_allow_html=True)

# --- Processing Loop ---
if st.session_state.running and video_path is not None:
    try:
        detector = load_model(MODEL_PATH)
        model_classes = list(detector.names.values()) if detector.names else []
        tracker = Tracker()
        rules = RuleEngine(
            crowd_threshold=CROWD_THRESHOLD,
            bag_stationary_seconds=BAG_STATIONARY_SECONDS,
            weapon_persist_frames=WEAPON_PERSIST_FRAMES,
            model_classes=model_classes
        )
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Could not open video source.")
    else:
        frame_idx = 0
        alert_history = []
        prev_time = time.time()

        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.info("Video finished.")
                break

            frame_idx += 1
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # Detection → Tracking → Rules
            detections = detector.detect(frame, conf_threshold=CONF_THRESHOLD)
            tracks = tracker.update(detections, frame)
            current_alerts = rules.process(tracks, frame_idx, frame_timestamp=time.time())

            # Update alert history
            for alert in current_alerts:
                alert_history.insert(0, {
                    "time": time.strftime('%H:%M:%S'),
                    "type": alert["type"],
                    "msg": alert["message"]
                })
            alert_history = alert_history[:15]

            # Draw overlays
            frame = draw_tracks(frame, tracks)
            frame = draw_alerts(frame, current_alerts)
            display_frame = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Update video
            video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

            # Update status panel
            status_placeholder.markdown(f"""
            <div class="stat-row"><span class="stat-label">Status</span><span class="badge-running">Running</span></div>
            <div class="stat-row"><span class="stat-label">Frame</span><span class="stat-value">{frame_idx}</span></div>
            <div class="stat-row"><span class="stat-label">FPS</span><span class="stat-value">{fps:.1f}</span></div>
            <div class="stat-row"><span class="stat-label">Active Tracks</span><span class="stat-value">{len(tracks)}</span></div>
            """, unsafe_allow_html=True)

            # Update alert log
            if alert_history:
                cards = ""
                for a in alert_history[:8]:
                    cards += f"""
                    <div class="alert-card">
                        <div class="alert-card-time">{a['time']} · {a['type']}</div>
                        <div class="alert-card-text">{a['msg']}</div>
                    </div>"""
                alert_placeholder.markdown(cards, unsafe_allow_html=True)
            else:
                alert_placeholder.markdown('<span style="font-size:13px; color:#9ca3af;">No alerts yet.</span>', unsafe_allow_html=True)

            # FPS cap
            process_time = time.time() - curr_time
            sleep_time = (1.0 / 30.0) - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    if source_type == "Upload Video" and video_path:
        os.remove(video_path)

elif st.session_state.running and video_path is None:
    st.error("Please select a video source first.")