import cv2
import time

def draw_tracks(frame, tracks):
    out = frame.copy()
    for t in tracks:
        try:
            ltrb = t.to_ltrb()
            x1,y1,x2,y2 = [int(v) for v in ltrb]
            tid = getattr(t, "track_id", None)
            name = t.get_det_class() if hasattr(t,"get_det_class") else "obj"
            cv2.rectangle(out, (x1,y1), (x2,y2), (255,200,0), 2)
            cv2.putText(out, f"ID:{tid} {name}", (x1, max(10,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)
        except Exception:
            pass
    return out

def draw_alerts(frame, alerts):
    out = frame.copy()
    y = 20
    for a in alerts[:5]:
        txt = f"{time.strftime('%H:%M:%S', time.localtime(a['timestamp']))} [{a['type']}] {a['message']}"
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y += 25
    return out
