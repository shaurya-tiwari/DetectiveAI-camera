import cv2
import numpy as np

def draw_detections(frame, detections):
    """
    Draw raw detections (before tracking) - mainly for debug.
    detections: [(x1, y1, x2, y2, conf, name), ...]
    """
    for (x1, y1, x2, y2, conf, name) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def draw_tracks(frame, tracks):
    """
    Draw tracked objects with IDs.
    tracks: list of Track objects
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # Get class name if available (stored in track.det_class or similar depending on implementation, 
        # but deep_sort_realtime stores it in track.get_det_class() or we passed it in update)
        # The 'name' passed to update_tracks is usually available via track.get_det_class() or track.det_class
        try:
            class_name = track.get_det_class()
        except:
            class_name = "obj"

        color = (255, 100, 0) # Blue-ish for tracks
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {class_name}", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_alerts(frame, alerts):
    """
    Draw alert overlays on the frame.
    alerts: list of alert dicts {type, message, bbox, ...}
    """
    for alert in alerts:
        bbox = alert.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            # Red box for alerts
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            label = f"ALERT: {alert['type']}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Also draw a list of alerts on top-right
    y_offset = 30
    for alert in alerts:
        text = f"{alert['type']}: {alert['message']}"
        cv2.putText(frame, text, (frame.shape[1] - 400, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
    return frame
