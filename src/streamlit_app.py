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
from telegramalert import TelegramNotifier


@st.cache_resource
def load_model(model_path):
    return Detector(model_path=model_path)


# --- Page Config ---
st.set_page_config(page_title="SentinelAI", layout="wide", page_icon="🎯")

# --- Custom CSS: Clean white modern light UI ---
st.markdown("""
<style>
/* Executive Dashboard (Navy & Gold) */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0b1121 !important;
    font-family: 'Outfit', sans-serif;
    color: #e2e8f0;
}

/* Hide default streamlit decoration */
[data-testid="stDecoration"], #MainMenu, footer { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161e31 !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { 
    font-family: 'Outfit', sans-serif !important; 
    color: #cbd5e1 !important;
}

/* Fix sidebar text color overrides */
[data-testid="stSidebar"] button * { color: #ffffff !important; }
[data-testid="stSidebar"] label { color: #e2e8f0 !important; }

/* Top header */
.sentinel-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    background: linear-gradient(145deg, #161e31, #0f1627);
    border-radius: 12px;
    border-bottom: 2px solid #d97706; /* Amber Gold accent */
    margin-bottom: 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
.sentinel-title {
    font-size: 24px;
    font-weight: 600;
    color: #f8fafc;
    margin: 0;
    letter-spacing: 0.02em;
}
.sentinel-sub {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
}

/* Section labels */
.section-label {
    font-size: 12px;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Alert card - Luxury Amber */
.alert-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
    border: 1px solid #334155;
    border-left: 4px solid #d97706;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
}
.alert-card-time {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 4px;
}
.alert-card-text { 
    font-size: 14px;
    font-weight: 600; 
    color: #fbbf24; 
}

/* Stat bar */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 12px 14px;
    background: #161e31;
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stat-label { color: #94a3b8; }
.stat-value { font-weight: 600; color: #f8fafc; }

/* Status badge */
.badge-running {
    display: inline-block;
    background: rgba(217, 119, 6, 0.15);
    color: #fbbf24;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 12px;
    border: 1px solid rgba(217, 119, 6, 0.3);
}
.badge-idle {
    display: inline-block;
    background: rgba(148, 163, 184, 0.1);
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
</style>
""", unsafe_allow_html=True)


# --- Config ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.onnx")
CONF_THRESHOLD = 0.35  # Base threshold — per-class thresholds in detection.py override this
CROWD_THRESHOLD = 20
BAG_STATIONARY_SECONDS = 5
WEAPON_PERSIST_FRAMES = 5  # Weapon must appear in 5 consecutive frames (~1s at 5fps) before alert

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

# Removed confidence slider for a zero-config experience. Using CONF_THRESHOLD backend setting.
if "running" not in st.session_state:
    st.session_state.running = False

if st.sidebar.button("▶  Start", use_container_width=True, type="primary"):
    st.session_state.running = True
if st.sidebar.button("⏹  Stop", use_container_width=True):
    st.session_state.running = False

st.sidebar.divider()
st.sidebar.caption("Model: `best.onnx` · Classes: person, armed man, gun, knife, baggage")
st.sidebar.caption(f"Persist: {WEAPON_PERSIST_FRAMES} frames")

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
            <span style="font-size:32px">+</span>
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
import logging
import queue
import threading
from collections import deque

# ── Production logger (replaces print statements) ───────────────
logger = logging.getLogger("SentinelAI")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)

# ── Model warmup helper ──────────────────────────────────────────
def warmup_model(detector, img_size=320):
    """Run 2 dummy frames through model so first real frame is fast."""
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(2):
        detector.detect(dummy, conf_threshold=0.5, img_size=img_size)
    logger.info("Model warmup complete.")


# --- Processing Loop (Production-grade) ---
if st.session_state.running and video_path is not None:

    # ── Initialize pipeline ──────────────────────────────────────
    try:
        detector = load_model(MODEL_PATH)

        # Pass model class names directly to RuleEngine
        # New model has clean classes: person, armed man, gun, knife, baggage
        model_classes = [c.lower() for c in detector.names.values()] if detector.names else []

        logger.info(f"Model loaded. Classes: {model_classes}")

        # Warmup: prevents first-frame latency spike
        warmup_model(detector, img_size=320)

        tracker = Tracker()
        rules = RuleEngine(
            crowd_threshold=CROWD_THRESHOLD,
            bag_stationary_seconds=BAG_STATIONARY_SECONDS,
            weapon_persist_frames=WEAPON_PERSIST_FRAMES,
            model_classes=model_classes
        )
        logger.info(f"RuleEngine: persons={rules.has_persons}, bags={rules.has_bags}, weapons={rules.has_weapons}")

    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")
        st.error(f"Failed to initialize: {e}")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video source.")
    else:
        # ── Shared state (thread-safe) ───────────────────────────
        frame_queue   = queue.Queue(maxsize=4)
        result_store  = {"tracks": [], "alerts": [], "latency": "—"}
        stop_event    = threading.Event()

        # Get native video FPS for throttling
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_delay = 1.0 / native_fps  # e.g. 25 FPS → sleep 0.04s between frames
        logger.info(f"Video native FPS: {native_fps:.1f} → frame_delay={frame_delay*1000:.0f}ms")

        fps_history   = deque(maxlen=15)
        alert_history = []
        frame_idx     = 0

        # ── INFERENCE THREAD ─────────────────────────────────────
        # Production pattern: runs AI in background, never blocks display
        def inference_worker():
            """
            Background thread: dequeues frames → runs AI → stores results.
            This is the standard CCTV AI pipeline pattern:
              - Decoupled from display thread
              - AI runs at its natural speed (6-7 FPS on Mac CPU)
              - Display runs at full capture speed
            """
            notifier = TelegramNotifier(min_strikes=1, cooldown_seconds=60)
            local_frame_idx = 0
            while not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                local_frame_idx += 1

                try:
                    dets    = detector.detect(frame, conf_threshold=CONF_THRESHOLD, img_size=320)
                    
                    # 📌 This new HuggingFace model only has classes like 'Gun', 'knife', 'grenade'
                    # No need to filter out 'person' or 'armed man' because this model doesn't even know what a person is!
                    weapon_dets = dets 
                            
                    tracks  = tracker.update(weapon_dets, frame)
                    alerts  = rules.process(tracks, local_frame_idx, frame_timestamp=time.time())

                    result_store["tracks"]  = tracks
                    result_store["alerts"]  = alerts
                    result_store["latency"] = detector.latency_stats().get("avg_ms", "—")

                    if alerts:
                        logger.warning(f"ALERT frame={local_frame_idx}: {[a['type'] for a in alerts]}")
                        for alert in alerts:
                            notifier.process_alert(alert, frame)
                            
                    if dets:
                        logger.info(f"DETECT frame={local_frame_idx}: {[(d[5], round(d[4],2)) for d in dets]}")

                except Exception as e:
                    logger.error(f"Inference error: {e}")

                frame_queue.task_done()

        # Start inference thread
        inf_thread = threading.Thread(target=inference_worker, daemon=True)
        inf_thread.start()
        logger.info("Inference thread started.")

        # ── CAPTURE + DISPLAY LOOP (Main thread) ─────────────────
        prev_time = time.time()
        frame_timer = time.time()  # For throttling to video speed

        try:
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.info("Video finished.")
                    break

                frame_idx += 1

                # ── THROTTLE to video's native FPS ───────────────────
                # Without this, display runs at 200+ FPS consuming frames instantly
                elapsed_since_last = time.time() - frame_timer
                sleep_needed = frame_delay - elapsed_since_last
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                frame_timer = time.time()

                # Push frame to inference queue (non-blocking drop)
                if not frame_queue.full():
                    frame_queue.put(frame.copy())

                # FPS: rolling average (smooth display)
                now = time.time()
                elapsed = now - prev_time
                prev_time = now
                if elapsed > 0:
                    fps_history.append(1.0 / elapsed)
                fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

                # Get latest AI results (non-blocking — use last known result)
                tracks  = result_store["tracks"]
                alerts  = result_store["alerts"]
                lat_str = f"{result_store['latency']} ms" if result_store['latency'] != '—' else "—"

                # Accumulate alert history
                for alert in alerts:
                    result_store["alerts"] = []  # consume alerts
                    alert_history.insert(0, {
                        "time": time.strftime('%H:%M:%S'),
                        "type": alert["type"],
                        "msg":  alert["message"]
                    })
                alert_history = alert_history[:20]

                # Draw overlays + display
                frame = draw_tracks(frame, tracks)
                frame = draw_alerts(frame, alerts)
                display_frame = cv2.resize(frame, (640, 360))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                # Status panel
                active = len([t for t in tracks if t.is_confirmed()])
                status_placeholder.markdown(f"""
                <div class="stat-row"><span class="stat-label">Status</span><span class="badge-running">Running</span></div>
                <div class="stat-row"><span class="stat-label">Frame</span><span class="stat-value">{frame_idx}</span></div>
                <div class="stat-row"><span class="stat-label">Display FPS</span><span class="stat-value">{fps:.1f}</span></div>
                <div class="stat-row"><span class="stat-label">AI Latency</span><span class="stat-value">{lat_str}</span></div>
                <div class="stat-row"><span class="stat-label">Active Tracks</span><span class="stat-value">{active}</span></div>
                """, unsafe_allow_html=True)

                # Alert log
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

        finally:
            # Graceful shutdown: signal thread to stop, wait for it
            stop_event.set()
            inf_thread.join(timeout=2.0)
            cap.release()
            logger.info("Pipeline shutdown complete.")
            if source_type == "Upload Video" and video_path and os.path.exists(video_path):
                os.remove(video_path)

elif st.session_state.running and video_path is None:
    st.error("Please select a video source first.")