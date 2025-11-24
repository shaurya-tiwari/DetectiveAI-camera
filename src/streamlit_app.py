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

# --- Sidebar Configuration ---
st.sidebar.header("System Configuration")

# Model Config
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
model_path = st.sidebar.text_input("Model Path", "models/best.pt")
if "yolov8" in model_path and "pt" in model_path and "weapon" not in model_path and "best" not in model_path:
    st.sidebar.info("NOTE: Standard YOLOv8 models (COCO) only detect 'knife' and 'scissors'. For 'gun' detection, you must train a custom model or download one.")

# Rule Config
st.sidebar.subheader("Rule Parameters")
crowd_threshold = st.sidebar.number_input("Crowd Threshold (persons)", min_value=1, value=20)
bag_stationary_seconds = st.sidebar.number_input("Bag Stationary Limit (s)", min_value=1, value=5)
weapon_persist_frames = st.sidebar.number_input("Weapon Persistence (frames)", min_value=1, value=5)

# Model Info & Remapping
st.sidebar.subheader("Model Classes")
try:
    detector = load_model(model_path)
    st.sidebar.write(f"Loaded {len(detector.names)} classes.")
    with st.sidebar.expander("Show Classes"):
        st.write(detector.names)
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

st.sidebar.subheader("Class Remapping")
remap_src = st.sidebar.text_input("Map Class (e.g. cell phone)", "")
remap_dst = st.sidebar.text_input("To Class (e.g. gun)", "gun")
use_remap = st.sidebar.checkbox("Enable Remapping")

# Input Source
st.sidebar.subheader("Video Source")
source_type = st.sidebar.radio("Select Source", ["Sample Video", "Upload Video", "Webcam"])

# Debug Options
st.sidebar.subheader("Debug Options")
simulate_gun = st.sidebar.checkbox("Simulate Gun with Cell Phone", help="Treat 'cell phone' as 'gun' to test alerts.")

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
        detector = load_model(model_path)
        
        # Apply remapping (reset first to avoid pollution from previous runs)
        detector.class_remap = {}
        if use_remap and remap_src and remap_dst:
            detector.class_remap[remap_src.lower()] = remap_dst.lower()
        if simulate_gun: # Keep legacy debug option working too
            detector.class_remap["cell phone"] = "gun"
            
        tracker = Tracker()
        rules = RuleEngine(
            crowd_threshold=crowd_threshold,
            bag_stationary_seconds=bag_stationary_seconds,
            weapon_persist_frames=weapon_persist_frames
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
        
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
            if not ret:
                st.info("Video finished.")
                break
            
            frame_idx += 1
            start_time = time.time()
            
            # 1. Detection
            detections = detector.detect(frame, conf_threshold=conf_threshold)
            
            # 2. Tracking
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
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Update UI
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            alert_placeholder.text_area("Recent Alerts", "\n".join(alert_history), height=300, key=f"alerts_{frame_idx}")
            
            status_placeholder.markdown(f"""
            **Status:** Running  
            **Frame:** {frame_idx}  
            **FPS:** {fps:.2f}  
            **Active Tracks:** {len(tracks)}
            """)
            
            # Small sleep to prevent UI freezing if processing is too fast (unlikely with deep learning)
            time.sleep(0.01)

    cap.release()
    if source_type == "Upload Video" and video_path:
        os.remove(video_path)

elif start_button and video_path is None:
    st.error("Please select a valid video source.")
