import numpy as np
import time

# ┌─────────────────────────────────────────────────────────────────────┐
# │                       DETECTION MODULE                              │
# │  Loads ONNX model → runs on each frame → returns bounding boxes    │
# │                                                                     │
# │  Model: best.onnx (YOLOv8n, trained on 77k CCTV images)            │
# │  Classes: person | armed man | gun | knife | baggage                │
# └─────────────────────────────────────────────────────────────────────┘

class Detector:

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  SETUP                                                           │
    # │  Load YOLO ONNX model + auto-pick CPU or GPU                    │
    # │                                                                  │
    # │  LATENCY TIPS:                                                   │
    # │    • GPU  → install onnxruntime-gpu  → ~4ms per frame            │
    # │    • CPU  → default (Apple/Intel)    → ~31ms per frame           │
    # │    • img_size=320 instead of 640     → ~2x faster               │
    # └──────────────────────────────────────────────────────────────────┘
    def __init__(self, model_path="models/best.onnx", device=None):
        try:
            from ultralytics import YOLO
        except Exception:
            raise RuntimeError("Install ultralytics: pip install ultralytics")

        import torch

        # Auto-pick best hardware: NVIDIA GPU (0), Apple Silicon (mps), or CPU (cpu)
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load ONNX model
        self.model = YOLO(model_path, task="detect")

        # Store class names from model — e.g. {0: 'person', 1: 'armed man', ...}
        self.names = getattr(self.model, "names", None)

        # ── Per-class confidence thresholds ──────────────────────────
        # Our model has 5 clean classes. Each threshold is tuned for CCTV:
        #   • "armed man" is the hardest — people fighting look like armed men → high threshold
        #   • "gun" and "knife" are precise → medium threshold
        #   • "person" and "baggage" are easy → lower threshold OK
        self.class_thresholds = {
            "person":    0.35,   # Easy class — stable across frames
            "armed man": 0.60,   # Very noisy in fight scenes → needs very high confidence
            "gun":       0.50,   # Raised: need clear visual to trigger, not just silhouette
            "knife":     0.52,   # Raised: knives are small → high conf = real detection
            "baggage":   0.40,   # Bags are distinctive — medium threshold
        }

        # Latency tracking — rolling average over last 30 frames
        self._latency_log = []

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  DETECT                                                          │
    # │  Takes one video frame → returns list of detected objects        │
    # │                                                                  │
    # │  Input:  frame (numpy image array from OpenCV)                   │
    # │  Output: [(x1, y1, x2, y2, confidence, "class_name"), ...]      │
    # └──────────────────────────────────────────────────────────────────┘
    def detect(self, frame, conf_threshold=0.35, img_size=640):

        # ── STEP 1: Run model inference ─────────────────────────────
        t0 = time.perf_counter()

        results = self.model.predict(
            frame,
            imgsz=img_size,
            conf=conf_threshold,   # Base confidence — per-class filter applied below
            device=self.device,
            verbose=False
        )

        # Track latency (rolling log of last 30 frames)
        inference_ms = (time.perf_counter() - t0) * 1000
        self._latency_log.append(inference_ms)
        if len(self._latency_log) > 30:
            self._latency_log.pop(0)

        if not results:
            return []

        r = results[0]
        detections = []

        # ── STEP 2: Parse bounding boxes ────────────────────────────
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                try:
                    cls_i = int(box.cls[0].item()) if hasattr(box.cls, "__len__") and len(box.cls) > 0 else int(box.cls.item())
                    conf  = float(box.conf[0].item()) if hasattr(box.conf, "__len__") and len(box.conf) > 0 else float(box.conf.item())

                    xyxy_tensor = box.xyxy[0] if box.xyxy.ndim > 1 else box.xyxy
                    xyxy = xyxy_tensor.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                except Exception:
                    continue

                # ── Resolve class index → class name ────────────────
                if isinstance(self.names, dict):
                    name = self.names.get(cls_i, str(cls_i))
                elif isinstance(self.names, (list, tuple)):
                    try:
                        name = self.names[int(cls_i)]
                    except Exception:
                        name = str(cls_i)
                else:
                    name = str(cls_i)

                name = name.lower().strip()

                # ── Skip tiny boxes (noise / shadows at CCTV distance) ──
                box_w = x2 - x1
                box_h = y2 - y1
                # Weapons at CCTV distance can be small — 15px minimum
                min_box = 15 if name in ("gun", "knife") else 20
                if box_w < min_box or box_h < min_box:
                    continue

                # ── Apply per-class confidence threshold ─────────────
                # Each class has a tuned threshold — overrides the base conf_threshold
                class_min_conf = self.class_thresholds.get(name, conf_threshold)
                if conf < class_min_conf:
                    continue

                detections.append((x1, y1, x2, y2, float(conf), name))

        # ── Agnostic NMS (Non-Maximum Suppression) ──────────────────
        # Lightweight pre-filter to remove obvious duplicate boxes of the same class
        def _iou(boxA, boxB):
            xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            union = boxAArea + boxBArea - interArea
            return interArea / float(union) if union > 0 else 0.0

        final_detections = []
        detections.sort(key=lambda x: x[4], reverse=True)   # Sort by confidence desc
        for d in detections:
            overlap = False
            for fd in final_detections:
                if d[5] == fd[5] and _iou(d, fd) > 0.45:
                    overlap = True
                    break
            if not overlap:
                final_detections.append(d)

        return final_detections

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  GET LATENCY STATS                                               │
    # │  Usage:  detector.latency_stats()                                │
    # │  Output: {"avg_ms": 31.3, "min_ms": 24.1, "device": "cpu"}      │
    # └──────────────────────────────────────────────────────────────────┘
    def latency_stats(self):
        if not self._latency_log:
            return {"avg_ms": None, "min_ms": None, "device": self.device}
        avg = sum(self._latency_log) / len(self._latency_log)
        mn  = min(self._latency_log)
        return {
            "avg_ms": round(avg, 1),
            "min_ms": round(mn, 1),
            "device": self.device,
        }
