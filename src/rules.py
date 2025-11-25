import time
import math

class RuleEngine:
    def __init__(self,
                 crowd_threshold=20,
                 bag_stationary_seconds=10,
                 weapon_persist_frames=4,
                 min_track_frames_before_bag=8,
                 alert_cooldowns=None):
        self.crowd_threshold = crowd_threshold
        self.bag_stationary_seconds = bag_stationary_seconds
        self.weapon_persist_frames = weapon_persist_frames
        self.min_track_frames_before_bag = min_track_frames_before_bag
        self.alert_cooldowns = alert_cooldowns or {"weapon":5.0, "bag":10.0, "crowd":10.0}
        self.bag_track_info = {}
        self.weapon_persist = {}
        self.last_alert_time = {"weapon":0.0, "bag":0.0, "crowd":0.0}
        self.track_frame_counts = {}
        self.last_weapon_alert_for_tid = {}

    def process(self, tracks, frame_index, frame_timestamp=None):
        now = frame_timestamp if frame_timestamp else time.time()
        alerts = []
        active_tids = set()
        persons_tracked = {}
        bags_tracked = {}
        weapon_candidates = {}

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            active_tids.add(tid)
            self.track_frame_counts[tid] = self.track_frame_counts.get(tid,0) + 1
            ltrb = t.to_ltrb()
            x1,y1,x2,y2 = int(ltrb[0]),int(ltrb[1]),int(ltrb[2]),int(ltrb[3])
            cx,cy = int((x1+x2)/2), int((y1+y2)/2)
            try:
                name = t.get_det_class().lower()
            except Exception:
                name = "unknown"
            if name == "person":
                persons_tracked[tid] = {"bbox":(x1,y1,x2,y2),"center":(cx,cy)}
            elif name in ["backpack","handbag","suitcase","bag"]:
                bags_tracked[tid] = {"bbox":(x1,y1,x2,y2),"center":(cx,cy)}
            elif any(k in name for k in ["firearm","gun","pistol","knife","weapon","scissors"]):
                weapon_candidates[tid] = {"bbox":(x1,y1,x2,y2),"center":(cx,cy),"name":name}

        # cleanup
        for tid in list(self.track_frame_counts.keys()):
            if tid not in active_tids:
                self.track_frame_counts.pop(tid,None)
                self.bag_track_info.pop(tid,None)
                self.weapon_persist.pop(tid,None)
                self.last_weapon_alert_for_tid.pop(tid,None)

        # crowd rule
        person_count = len(persons_tracked)
        if person_count >= self.crowd_threshold:
            if now - self.last_alert_time["crowd"] >= self.alert_cooldowns["crowd"]:
                alerts.append({"type":"CROWD","message":f"Crowd detected: {person_count} people","timestamp":now,"frame_idx":frame_index})
                self.last_alert_time["crowd"] = now

        # bag rule
        for tid, info in bags_tracked.items():
            if self.track_frame_counts.get(tid,0) < self.min_track_frames_before_bag:
                continue
            cx,cy = info["center"]
            if tid not in self.bag_track_info:
                self.bag_track_info[tid] = {"last_center":(cx,cy),"first_seen":now,"last_seen":now}
                continue
            entry = self.bag_track_info[tid]
            dx = entry["last_center"][0] - cx
            dy = entry["last_center"][1] - cy
            dist = math.hypot(dx,dy)
            if dist > 10:
                entry["last_center"] = (cx,cy)
                entry["first_seen"] = now
                entry["last_seen"] = now
            else:
                entry["last_seen"] = now
            duration = entry["last_seen"] - entry["first_seen"]
            has_owner = False
            for pid,pinfo in persons_tracked.items():
                pcx,pcy = pinfo["center"]
                p_dist = math.hypot(pcx - cx, pcy - cy)
                if p_dist < 150:
                    has_owner = True
                    break
            if not has_owner and duration >= self.bag_stationary_seconds:
                if now - self.last_alert_time["bag"] >= self.alert_cooldowns["bag"]:
                    alerts.append({"type":"UNATTENDED_BAG","message":f"Bag {tid} unattended for {duration:.1f}s","timestamp":now,"frame_idx":frame_index,"bbox":info["bbox"],"track_id":tid})
                    self.last_alert_time["bag"] = now
                    entry["first_seen"] = now

        # weapon rule
        for tid,winfo in weapon_candidates.items():
            prev = self.weapon_persist.get(tid,{"count":0})
            prev["count"] = prev.get("count",0) + 1
            self.weapon_persist[tid] = prev
            if prev["count"] >= self.weapon_persist_frames:
                last_tid_time = self.last_weapon_alert_for_tid.get(tid,0.0)
                if now - last_tid_time >= self.alert_cooldowns["weapon"] and now - self.last_alert_time["weapon"] >= self.alert_cooldowns["weapon"]:
                    alerts.append({"type":"WEAPON","message":f"Weapon ({winfo['name']}) detected!","timestamp":now,"frame_idx":frame_index,"bbox":winfo["bbox"],"track_id":tid})
                    self.last_alert_time["weapon"] = now
                    self.last_weapon_alert_for_tid[tid] = now

        return alerts
