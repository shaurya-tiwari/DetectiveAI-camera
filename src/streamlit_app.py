import streamlit as st
import cv2
import tempfile
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Add src to path so we can import modules if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection import Detector
from src.tracking import Tracker
from src.rules import RuleEngine
from src.rules import RuleEngine
from src.visualize import draw_tracks, draw_detections, draw_alerts

@st.cache_resource
def load_model(model_path):
    return Detector(model_path=model_path)

st.set_page_config(page_title="SentinelAI Dashboard", layout="wide")

st.title("SentinelAI: Intelligent Surveillance System")
st.markdown("Real-time anomaly detection: Weapons, Unattended Bags, and Crowds.")

# --- Configuration ---
# Hardcoded optimal settings
MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.40
CROWD_THRESHOLD = 20
BAG_STATIONARY_SECONDS = 5
WEAPON_PERSIST_FRAMES = 5

# --- Sidebar ---
st.sidebar.header("Video Source")
source_type = st.sidebar.radio("Select Source", ["Sample Video", "Upload Video", "Webcam"])

video_path = None
if source_type == "Sample Video":
    # List videos in 'videos' folder if exists
    video_folder = os.path.join(os.path.dirname(__file__), "..", "videos")
    if os.path.exists(video_folder):
        files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        if files:
            selected_file = st.sidebar.selectbox("Choose a video", files)
            video_path = os.path.join(video_folder, selected_file)
        else:
            st.sidebar.warning("No videos found in 'videos' folder.")
    else:
        st.sidebar.warning("'videos' folder not found.")
elif source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif source_type == "Webcam":
    video_path = 0 # Webcam index

# Controls
start_button = st.sidebar.button("Start Surveillance", type="primary")
stop_button = st.sidebar.button("Stop")

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Alert Log")
    alert_placeholder = st.empty()
    
    st.subheader("System Status")
    status_placeholder = st.empty()

# --- Processing Loop ---
if start_button and video_path is not None:
    # Initialize Modules
    try:
        detector = load_model(MODEL_PATH)
        
        # Reset remapping just in case, though we don't use it anymore
        detector.class_remap = {}
            
        tracker = Tracker()
        rules = RuleEngine(
            crowd_threshold=CROWD_THRESHOLD,
            bag_stationary_seconds=BAG_STATIONARY_SECONDS,
            weapon_persist_frames=WEAPON_PERSIST_FRAMES
        )
    except Exception as e:
        st.error(f"Failed to initialize modules: {e}")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video source.")
    else:
        frame_idx = 0
        alert_history = []
        prev_time = time.time()
        
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
            if not ret:
                st.info("Video finished.")
                break
            
            frame_idx += 1
            
            # Calculate FPS based on time since last frame
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time
            
            # Frame Skipping Logic
            FRAME_SKIP = 2
            
            # 1. Detection
            if frame_idx % FRAME_SKIP == 0:
                detections = detector.detect(frame, conf_threshold=CONF_THRESHOLD)
            else:
                detections = [] # Skip detection
            
            # 2. Tracking
            # Update tracker. If detections are empty (skipped frame), it will predict.
            tracks = tracker.update(detections, frame)
            
            # 3. Rules
            current_alerts = rules.process(tracks, frame_idx, frame_timestamp=time.time())
            
            # Update Alert History
            if current_alerts:
                for alert in current_alerts:
                    alert_history.insert(0, f"{time.strftime('%H:%M:%S')} - [{alert['type']}] {alert['message']}")
                # Keep last 20
                alert_history = alert_history[:20]
            
            # 4. Visualization
            # Draw tracks
            frame = draw_tracks(frame, tracks)
            # Draw alerts
            frame = draw_alerts(frame, current_alerts)
            
            # Update UI
            # Resize for faster transmission to browser
            display_frame = cv2.resize(frame, (640, 360))
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            alert_placeholder.text_area("Recent Alerts", "\n".join(alert_history), height=300, key=f"alerts_{frame_idx}")
            
            status_placeholder.markdown(f"""
            **Status:** Running  
            **Frame:** {frame_idx}  
            **FPS:** {fps:.2f}  
            **Active Tracks:** {len(tracks)}
            """)
            
            # FPS Capping (Target 30 FPS)
            process_time = time.time() - curr_time
            target_time = 1.0 / 30.0
            if process_time < target_time:
                time.sleep(target_time - process_time)

    cap.release()
    if source_type == "Upload Video" and video_path:
        os.remove(video_path)

elif start_button and video_path is None:
    st.error("Please select a valid video source.")