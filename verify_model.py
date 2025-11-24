import cv2
import numpy as np
from ultralytics import YOLO
import os

def verify_model(model_path="models/best.pt"):
    print(f"Verifying model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        
        print("Model Classes:")
        print(model.names)
        
        # Create a dummy black image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("Running inference on dummy image...")
        results = model.predict(img, verbose=False)
        print("Inference successful.")
        
        if results:
            print(f"Results object type: {type(results)}")
            print(f"Number of detections: {len(results[0].boxes)}")
            
    except Exception as e:
        print(f"Error verifying model: {e}")

if __name__ == "__main__":
    verify_model()
