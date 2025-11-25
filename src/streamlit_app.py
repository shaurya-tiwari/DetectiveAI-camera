import streamlit as st
import cv2
import tempfile
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from huggingface_hub import hf_hub_download

# Add src to path so we can import modules if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection import Detector
from src.tracking import Tracker
from src.rules import RuleEngine
from src.visualize import draw_tracks, draw_detections, draw_alerts

@st.cache_resource
def load_model(model_path):
    return Detector(model_path=model_path)

@st.cache_resource
def download_firearm_model():
    """Download the firearm detection model from Hugging Face"""
    with st.spinner("Downloading firearm detection model from Hugging Face..."):
        try:
            model_path = hf_hub_download(
                repo_id="Subh775/Firearm_Detection_Yolov8n",
                filename="weights/best.pt"
            )
            return model_path
        except Exception as e:
            st.error(f"Failed to download firearm model: {e}")
            return None
model_choice = st.sidebar.radio(
    "Select Model",
    ["Firearm Detection (HuggingFace)", "Local Model (COCO)"],
    index=0
)
USE_FIREARM_MODEL = (model_choice == "Firearm Detection (HuggingFace)")

st.sidebar.header("📹 Video Source")
source_type = st.sidebar.radio("Select Source", ["Sample Video", "Upload Video", "Webcam"])

# Show model info
with st.sidebar.expander("🔍 Model Info & Settings"):
    if USE_FIREARM_MODEL:
        st.success("✅ Using Firearm Detection Model")
        st.write("- **Source:** Subh775/Firearm_Detection_Yolov8n")
        st.write("- **Classes:** Gun")
        st.write("- **Accuracy:** 89% mAP@0.5")
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
    video_path = 0

# Controls
start_button = st.sidebar.button("▶️ Start Surveillance", type="primary")
stop_button = st.sidebar.button("⏹️ Stop")

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("🚨 Alert Log")
    alert_placeholder = st.empty()
    
    st.subheader("📊 System Status")
    status_placeholder = st.empty()
    
    st.subheader("🔍 Live Detections")
    detection_placeholder = st.empty()

# --- Processing Loop ---
if start_button and video_path is not None:
    # Initialize Modules
    try:
        # Load appropriate model
        if USE_FIREARM_MODEL:
            firearm_model_path = download_firearm_model()
            if firearm_model_path is None:
                st.error("Failed to download firearm model. Please check your internet connection.")
                st.stop()
            MODEL_PATH = firearm_model_path
            st.success("✅ Firearm detection model loaded!")
        else:
            MODEL_PATH = LOCAL_MODEL_PATH
            st.info("ℹ️ Using local COCO model (no gun detection)")
        
        detector = load_model(MODEL_PATH)
        
        # Check model capabilities
        has_gun = 'gun' in [str(v).lower() for v in detector.names.values()]
        has_knife = 'knife' in [str(v).lower() for v in detector.names.values()]
        
        # No class remapping needed for firearm model!
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
        detection_log = []
        
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
                    
                    # Highlight weapons in detection log
                    if class_name.lower() in ['knife', 'gun', 'weapon', 'scissors']:
                        detection_log.append(f"🚨 WEAPON: {class_name}")
            else:
                detections = []
            
            # 2. Tracking
            tracks = tracker.update(detections, frame)
            
            # 3. Rules
            current_alerts = rules.process(tracks, frame_idx, frame_timestamp=time.time())
            
            # Update Alert History
            if current_alerts:
                for alert in current_alerts:
                    alert_history.insert(0, f"{time.strftime('%H:%M:%S')} - [{alert['type']}] {alert['message']}")
                alert_history = alert_history[:20]
            
            # 4. Visualization
            frame = draw_tracks(frame, tracks)
            frame = draw_alerts(frame, current_alerts)
            
            # Update UI
            display_frame = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            alert_placeholder.text_area("Recent Alerts", "\n".join(alert_history) if alert_history else "No alerts yet", height=250, key=f"alerts_{frame_idx}")
            
            # Show live detections
            if detection_log:
                detection_placeholder.text_area("Detections", "\n".join(detection_log[:10]), height=150, key=f"det_{frame_idx}")
            else:
                detection_placeholder.text_area("Detections", "No detections", height=150, key=f"det_{frame_idx}")
            
            status_placeholder.markdown(f"""
            **Status:** 🟢 Running  
            **Frame:** {frame_idx}  
            **FPS:** {fps:.2f}  
            **Active Tracks:** {len(tracks)}  
            **Detections:** {len(detections)}
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