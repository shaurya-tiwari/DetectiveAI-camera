import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.class_remap = None

    def detect(self, frame, conf_threshold=0.25, img_size=640):
        """
        Run YOLOv8 detection on the frame.
        Returns a list of detections: [(x1, y1, x2, y2, conf, class_name), ...]
        """
        results = self.model.predict(frame, imgsz=img_size, conf=conf_threshold, verbose=False)
        r = results[0]
        
        detections = []
        if hasattr(r, "boxes"):
            for box in r.boxes:
                try:
                    cls_i = int(box.cls.cpu().numpy())
                    conf = float(box.conf.cpu().numpy())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                except:
                    # Fallback for different tensor types if needed
                    cls_i = int(box.cls.numpy())
                    conf = float(box.conf.numpy())
                    xyxy = box.xyxy[0].numpy().astype(int)
                
                x1, y1, x2, y2 = xyxy.tolist()
                name = self.names.get(cls_i, str(cls_i)) if isinstance(self.names, dict) else self.names[cls_i]
                name = name.lower()
                
                # Remap if needed (e.g. cell phone -> gun for testing)
                if hasattr(self, "class_remap") and self.class_remap and name in self.class_remap:
                    name = self.class_remap[name]
                
                detections.append((x1, y1, x2, y2, conf, name))
        
        return detections
