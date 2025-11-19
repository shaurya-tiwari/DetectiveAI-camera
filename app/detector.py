# app/detector.py
import os
import time
import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from math import hypot

# --- simple helper to save snapshot ---
def save_snapshot(img, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{tag}_{int(time.time())}.jpg")
    cv2.imwrite(fname, img)
    return fname

def center_from_xyxy(x1,y1,x2,y2):
    return int((x1+x2)/2), int((y1+y2)/2)

def list_video_files(path):
    if os.path.isdir(path):
        exts = (".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg")
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(exts)])
        return files
    elif os.path.isfile(path):
        return [path]
    else:
        return []

def main(args):
    # Config
    VIDEO_SOURCE = args.source
    OUTPUT_DIR = args.output
    MODEL_PATH = args.model if args.model else "yolov8n.pt"

    IMG_SIZE = args.imgsz
    CONF_THR = args.conf
    PERSON_CONF_COUNT = args.person_conf
    CROWD_THRESHOLD = args.crowd_threshold
    BAG_STATIONARY_SECONDS = args.bag_seconds
    WEAPON_PERSIST_FRAMES = args.weapon_persist
    WEAPON_CONF_MIN = args.weapon_conf
    SAVE_SNAPSHOTS = True

    ALERT_COOLDOWN = {"weapon": args.cool_weapon, "bag": args.cool_bag, "crowd": args.cool_crowd}
    MIN_TRACK_FRAMES_BEFORE_BAG = args.min_track_frames

    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Model loaded. Names:", model.names)

    tracker = DeepSort(max_age=30)

    video_files = list_video_files(VIDEO_SOURCE)
    if not video_files:
        print("No videos found in", VIDEO_SOURCE)
        return
    print("Found videos:", video_files)

    bag_track_info = {}
    weapon_persist = {}
    last_alert_time = {"weapon": 0.0, "bag": 0.0, "crowd": 0.0}
    last_weapon_alert_for_tid = {}
    track_frame_counts = {}

    for vf in video_files:
        print("\n--- SCANNING:", vf)
        cap = cv2.VideoCapture(vf)
        if not cap.isOpened():
            print("Cannot open", vf)
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THR, verbose=False)
            r = results[0]
            class_names = model.names

            detections = []
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    try:
                        cls_i = int(box.cls.cpu().numpy())
                        conf = float(box.conf.cpu().numpy())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    except:
                        cls_i = int(box.cls.numpy())
                        conf = float(box.conf.numpy())
                        xyxy = box.xyxy[0].numpy().astype(int)
                    x1,y1,x2,y2 = xyxy.tolist()
                    name = class_names.get(cls_i, str(cls_i)) if isinstance(class_names, dict) else class_names[cls_i]
                    detections.append((x1,y1,x2,y2,conf,name.lower()))

            ds_input = [((x1,y1,x2,y2), conf, name) for (x1,y1,x2,y2,conf,name) in detections]
            tracks = tracker.update_tracks(ds_input, frame=frame)

            persons_tracked = {}
            bags_tracked = {}
            weapon_candidates_tracked = {}
            active_tids = set()

            # centroid-match detection names to tracks and update frame counts
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                active_tids.add(tid)
                x1,y1,x2,y2 = [int(v) for v in t.to_tlbr()]
                cx, cy = center_from_xyxy(x1,y1,x2,y2)
                track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

                # find nearest detection by centroid
                nearest_name, nearest_conf = None, 0.0
                nearest_dist = 1e9
                for (bx1,by1,bx2,by2,conf,name) in detections:
                    bcx,bcy = (bx1+bx2)/2, (by1+by2)/2
                    d = (bcx-cx)**2 + (bcy-cy)**2
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_name = name
                        nearest_conf = conf

                name = nearest_name or "unknown"

                if name == "person" and nearest_conf >= PERSON_CONF_COUNT:
                    persons_tracked[tid] = (x1,y1,x2,y2,nearest_conf,(cx,cy))
                elif name in ("backpack","handbag","suitcase","bag","briefcase"):
                    bags_tracked[tid] = (x1,y1,x2,y2,nearest_conf,(cx,cy))
                else:
                    if nearest_conf >= WEAPON_CONF_MIN:
                        weapon_candidates_tracked[tid] = (x1,y1,x2,y2,nearest_conf,(cx,cy),name)

            # cleanup frame counts for tracks that disappeared
            for tid in list(track_frame_counts.keys()):
                if tid not in active_tids:
                    track_frame_counts.pop(tid, None)

            # CROWD alert with cooldown
            person_count = len(persons_tracked)
            now = time.time()
            if person_count >= CROWD_THRESHOLD and (now - last_alert_time["crowd"]) >= ALERT_COOLDOWN["crowd"]:
                print(f"[ALERT][CROWD] {person_count} persons detected in {os.path.basename(vf)} (frame {frame_idx})")
                if SAVE_SNAPSHOTS:
                    save_snapshot(frame, OUTPUT_DIR, f"alert_crowd_{frame_idx}")
                last_alert_time["crowd"] = now

            # BAG stationary detection (require min frames and cooldown)
            for tid, info in list(bags_tracked.items()):
                if track_frame_counts.get(tid, 0) < MIN_TRACK_FRAMES_BEFORE_BAG:
                    continue
                cx,cy = info[5]
                if tid not in bag_track_info:
                    bag_track_info[tid] = {"last_center":(cx,cy), "first_seen":now, "last_seen":now}
                else:
                    entry = bag_track_info[tid]
                    dx = entry["last_center"][0] - cx
                    dy = entry["last_center"][1] - cy
                    dist = (dx*dx + dy*dy)**0.5
                    if dist <= 10:
                        entry["last_seen"] = now
                    else:
                        entry["last_center"] = (cx,cy)
                        entry["last_seen"] = now
                    duration = entry["last_seen"] - entry["first_seen"]
                    if duration >= BAG_STATIONARY_SECONDS and (now - last_alert_time["bag"]) >= ALERT_COOLDOWN["bag"]:
                        print(f"[ALERT][BAG-STATIONARY] bag track {tid} stationary for {duration:.1f}s in {os.path.basename(vf)} (frame {frame_idx})")
                        if SAVE_SNAPSHOTS:
                            save_snapshot(frame, OUTPUT_DIR, f"alert_bag_{tid}_{frame_idx}")
                        last_alert_time["bag"] = now
                        try:
                            del bag_track_info[tid]
                        except KeyError:
                            pass

            # WEAPON heuristics: proximity + persistence + per-track cooldown
            for tid, wc in list(weapon_candidates_tracked.items()):
                wx,wy = wc[5]
                near_person = False
                for pid,pinfo in persons_tracked.items():
                    pcx,pcy = pinfo[5]
                    p_w = max(1, pinfo[2] - pinfo[0])
                    dist = ((pcx-wx)**2 + (pcy-wy)**2)**0.5
                    if dist <= max(40, 0.9 * p_w):
                        near_person = True
                        break

                if near_person:
                    prev = weapon_persist.get(tid, {"center":(wx,wy), "count":0})
                    dmove = ((prev["center"][0]-wx)**2 + (prev["center"][1]-wy)**2)**0.5
                    if dmove <= max(30, IMG_SIZE*0.02):
                        prev["count"] += 1
                    else:
                        prev["count"] = 1
                        prev["center"] = (wx,wy)
                    weapon_persist[tid] = prev

                    if prev["count"] >= WEAPON_PERSIST_FRAMES:
                        last_tid_time = last_weapon_alert_for_tid.get(tid, 0.0)
                        if (now - last_tid_time) >= ALERT_COOLDOWN["weapon"] and (now - last_alert_time["weapon"]) >= ALERT_COOLDOWN["weapon"]:
                            print(f"[ALERT][WEAPON] possible weapon near person (track {tid}) in {os.path.basename(vf)} (frame {frame_idx})")
                            if SAVE_SNAPSHOTS:
                                save_snapshot(frame, OUTPUT_DIR, f"alert_weapon_{tid}_{frame_idx}")
                            last_weapon_alert_for_tid[tid] = now
                            last_alert_time["weapon"] = now
                        weapon_persist[tid] = {"center":(wx,wy), "count":0}
                else:
                    weapon_persist.pop(tid, None)

        cap.release()

    print("Scan complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="path to weights (default: use yolov8n.pt)")
    parser.add_argument("--source", default="../videos", help="video file or folder")
    parser.add_argument("--output", default="../outputs", help="output folder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--person_conf", type=float, default=0.50)
    parser.add_argument("--crowd_threshold", type=int, default=20)
    parser.add_argument("--bag_seconds", type=int, default=8)
    parser.add_argument("--weapon_persist", type=int, default=5)
    parser.add_argument("--weapon_conf", type=float, default=0.25)
    parser.add_argument("--cool_weapon", type=int, default=8)
    parser.add_argument("--cool_bag", type=int, default=15)
    parser.add_argument("--cool_crowd", type=int, default=10)
    parser.add_argument("--min_track_frames", type=int, default=8)

    args = parser.parse_args()
    main(args)
