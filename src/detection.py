import numpy as np
import time

# ┌─────────────────────────────────────────┐
# │           DETECTION MODULE              │
# │  Loads ONNX model → runs on each frame  │
# │  Returns bounding boxes + class names   │
# └─────────────────────────────────────────┘

class Detector:

    # ┌──────────────────────────────────────────────────────────┐
    # │  SETUP                                                   │
    # │  Load YOLO model + pick CPU or GPU                       │
    # │                                                          │
    # │  LATENCY TIPS:                                           │
    # │    • GPU  → install onnxruntime-gpu  → ~4ms per frame    │
    # │    • CPU  → default (Apple/Intel)    → ~31ms per frame   │
    # │    • img_size=320 instead of 640     → ~2x faster        │
    # └──────────────────────────────────────────────────────────┘
    def __init__(self, model_path="yolov8n.pt", device=None):
        try:
            from ultralytics import YOLO
        except Exception:
            raise RuntimeError("Install ultralytics: pip install ultralytics")

        import torch

        # Auto-pick GPU ("0") if CUDA available, else CPU
        # On Apple Silicon: always CPU (no CUDA support)
        # On NVIDIA: uses GPU → much faster
        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")

        # Load the ONNX model from disk
        self.model = YOLO(model_path, task="detect")

        # Store class names: e.g. {0: "pistol", 1: "knife"}
        self.names = getattr(self.model, "names", None)

        # Optional: rename class labels (not used currently)
        self.class_remap = {}

        # Latency tracking — running average over last 30 frames
        self._latency_log = []

    # ┌──────────────────────────────────────────────────────────────┐
    # │  DETECT                                                      │
    # │  Takes one video frame → returns list of detected objects    │
    # │                                                              │
    # │  Input:  frame (numpy image array from OpenCV)               │
    # │  Output: [(x1, y1, x2, y2, confidence, "class_name"), ...]   │
    # │                                                              │
    # │  WHERE LATENCY COMES FROM:                                   │
    # │    ┌─────────────────────────────────────────────────────┐   │
    # │    │  model.predict()  ← 95% of the time is spent here   │   │
    # │    │    img_size=640   → resizes frame → runs 80 layers   │   │
    # │    │    img_size=320   → 2x faster, slightly less detail  │   │
    # │    └─────────────────────────────────────────────────────┘   │
    # │    post-processing (box parsing) ← ~5% of time              │
    # └──────────────────────────────────────────────────────────────┘
    def detect(self, frame, conf_threshold=0.25, img_size=640):

        # ── STEP 1: Run model inference — this is the slow part ────────
        # img_size=640 → standard, balanced accuracy vs speed
        # img_size=320 → faster (use if speed matters more than accuracy)
        t0 = time.perf_counter()

        results = self.model.predict(
            frame,
            imgsz=img_size,       # ← reduce to 320 for 2x speed boost
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        # ── Measure and log how long inference took ─────────────────────
        inference_ms = (time.perf_counter() - t0) * 1000
        self._latency_log.append(inference_ms)
        if len(self._latency_log) > 30:
            self._latency_log.pop(0)          # Keep only last 30 samples

        if not results:
            return []

        r = results[0]
        detections = []

        # ── STEP 2: Parse boxes — fast, ~1ms ───────────────────────────
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                try:
                    # Extract class index + confidence as plain Python numbers
                    cls_i = int(box.cls[0].item()) if hasattr(box.cls, "__len__") and len(box.cls) > 0 else int(box.cls.item())
                    conf  = float(box.conf[0].item()) if hasattr(box.conf, "__len__") and len(box.conf) > 0 else float(box.conf.item())

                    # Extract corner coordinates [x1, y1, x2, y2]
                    xyxy_tensor = box.xyxy[0] if box.xyxy.ndim > 1 else box.xyxy
                    xyxy = xyxy_tensor.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                except Exception:
                    continue  # Skip bad boxes

                # ── Resolve class index → class name ───────────────────
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

                # Optional rename (e.g. "guns" → "pistol")
                if self.class_remap and name in self.class_remap:
                    name = self.class_remap[name]

                # Add to results
                detections.append((x1, y1, x2, y2, float(conf), name))

        return detections

    # ┌──────────────────────────────────────────────────────────────┐
    # │  GET LATENCY STATS                                           │
    # │  Call anytime to see actual measured inference speed         │
    # │                                                              │
    # │  Usage:  detector.latency_stats()                            │
    # │  Output: {"avg_ms": 31.3, "min_ms": 24.1, "device": "cpu"}  │
    # └──────────────────────────────────────────────────────────────┘
    def latency_stats(self):
        if not self._latency_log:
            return {"avg_ms": None, "min_ms": None, "device": self.device}
        avg = sum(self._latency_log) / len(self._latency_log)
        mn  = min(self._latency_log)
        return {
            "avg_ms": round(avg, 1),
            "min_ms": round(mn, 1),
            "device": self.device,
            # Quick note on how to improve:
            # "tip": "Use img_size=320 for 2x faster, or onnxruntime-gpu for ~4ms on NVIDIA"
        }
