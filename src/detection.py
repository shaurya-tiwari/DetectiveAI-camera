import numpy as np

class Detector:
    def __init__(self, model_path="yolov8n.pt", device=None):
        try:
            from ultralytics import YOLO
        except Exception:
            raise RuntimeError("Install ultralytics: pip install ultralytics")
        import torch
        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path, task="detect")
        self.names = getattr(self.model, "names", None)
        self.class_remap = {}

    def detect(self, frame, conf_threshold=0.25, img_size=640):
        """
        Returns list of (x1,y1,x2,y2,conf,class_name)
        """
        results = self.model.predict(frame, imgsz=img_size, conf=conf_threshold, device=self.device, verbose=False)
        if not results:
            return []
        r = results[0]
        detections = []
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                try:
                    # Extract scalar values safely using PyTorch .item()
                    cls_i = int(box.cls[0].item()) if hasattr(box.cls, "__len__") and len(box.cls) > 0 else int(box.cls.item())
                    conf = float(box.conf[0].item()) if hasattr(box.conf, "__len__") and len(box.conf) > 0 else float(box.conf.item())
                    
                    # Extract box coordinates xyxy
                    xyxy_tensor = box.xyxy[0] if box.xyxy.ndim > 1 else box.xyxy
                    xyxy = xyxy_tensor.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                except Exception:
                    continue

                # resolve class name
                if isinstance(self.names, dict):
                    name = self.names.get(cls_i, str(cls_i))
                elif isinstance(self.names, (list, tuple)):
                    try:
                        name = self.names[int(cls_i)]
                    except Exception:
                        name = str(cls_i)
                else:
                    name = str(cls_i)

                name = name.lower()
                if self.class_remap and name in self.class_remap:
                    name = self.class_remap[name]

                detections.append((x1, y1, x2, y2, float(conf), name))
        return detections
