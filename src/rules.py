import time
import math

# ┌──────────────────────────────────────────────────────────────────┐
# │                      RULES ENGINE                                │
# │  Reads confirmed tracks each frame → decides what is dangerous   │
# │                                                                  │
# │  3 rules:                                                        │
# │    WEAPON  → pistol/knife seen for N consecutive frames          │
# │    CROWD   → too many people in frame at once                    │
# │    BAG     → bag left alone (no person nearby) for N seconds     │
# │                                                                  │
# │  Rules auto-enable/disable based on what the model can detect    │
# └──────────────────────────────────────────────────────────────────┘

# Keyword lists to classify object names into categories
PERSON_KEYWORDS = ["person", "people", "human"]
BAG_KEYWORDS    = ["backpack", "handbag", "suitcase", "bag", "luggage"]
WEAPON_KEYWORDS = ["firearm", "gun", "pistol", "knife", "weapon", "scissors", "rifle"]


class RuleEngine:

    # ┌──────────────────────────────────────────────────────────────┐
    # │  SETUP                                                       │
    # │  Receives model class names → auto-enables matching rules    │
    # └──────────────────────────────────────────────────────────────┘
    def __init__(self,
                 crowd_threshold=20,
                 bag_stationary_seconds=10,
                 weapon_persist_frames=4,
                 min_track_frames_before_bag=8,
                 alert_cooldowns=None,
                 model_classes=None):

        # 📌 KEY CONCEPT: Instance Variables (Object State)
        # All thresholds stored as instance variables so they can be tuned at runtime without changing code
        # Thresholds
        self.crowd_threshold           = crowd_threshold
        self.bag_stationary_seconds    = bag_stationary_seconds
        self.weapon_persist_frames     = weapon_persist_frames
        self.min_track_frames_before_bag = min_track_frames_before_bag

        # Cooldowns: seconds between repeated alerts of same type
        self.alert_cooldowns = alert_cooldowns or {"weapon": 5.0, "bag": 10.0, "crowd": 10.0}

        # 📌 KEY CONCEPT: Multiple HashMaps for Per-Object State Tracking
        # Each dict is a separate 'track_id → state' mapping — this is the system's memory across frames
        # ── State memory across frames ────────────────────────────
        self.bag_track_info           = {}  # tracks bag positions + timers
        self.weapon_persist           = {}  # counts consecutive frames weapon seen
        self.last_alert_time          = {"weapon": 0.0, "bag": 0.0, "crowd": 0.0}
        self.track_frame_counts       = {}  # how many frames each track has been seen
        self.last_weapon_alert_for_tid = {}  # per-track weapon alert cooldown

        # 📌 KEY CONCEPT: List Comprehension + any() for Keyword Matching
        # Auto-detects which rules to enable based on what class names the loaded model supports
        # ── Check which rules are relevant for this model ────────
        classes = [c.lower() for c in (model_classes or [])]
        self.has_persons = any(any(k in c for k in PERSON_KEYWORDS) for c in classes)
        self.has_bags    = any(any(k in c for k in BAG_KEYWORDS)    for c in classes)
        self.has_weapons = any(any(k in c for k in WEAPON_KEYWORDS) for c in classes)

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  CLASSIFY                                                        │
    # │  Maps a class name string → category (person / bag / weapon)     │
    # └──────────────────────────────────────────────────────────────────┘
    def _classify(self, name):
        # 📌 KEY CONCEPT: Short-circuit Evaluation with any()
        # any() stops at the first match — O(N) worst case but fast in practice for small keyword lists
        if any(k in name for k in PERSON_KEYWORDS): return "person"
        if any(k in name for k in BAG_KEYWORDS):    return "bag"
        if any(k in name for k in WEAPON_KEYWORDS): return "weapon"
        return None

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  PROCESS                                                         │
    # │  Main function — called every frame with the current track list  │
    # │                                                                  │
    # │  Input:  tracks (list of track objects from tracker)             │
    # │  Output: list of alert dicts [{"type", "message", ...}, ...]     │
    # └──────────────────────────────────────────────────────────────────┘
    def process(self, tracks, frame_index, frame_timestamp=None):
        now = frame_timestamp if frame_timestamp else time.time()
        alerts = []

        # 📌 KEY CONCEPT: Set for O(1) Membership Lookup
        # Using a Set instead of a List — 'tid not in active_tids' is O(1) vs O(N) for a list
        active_tids      = set()

        # 📌 KEY CONCEPT: Separate HashMaps per Category
        # Splitting into 3 dicts lets us count persons with len(), loop bags separately, etc.
        persons_tracked  = {}
        bags_tracked     = {}
        weapon_candidates = {}

        # ── Step 1: Sort all confirmed tracks into categories ────────
        for t in tracks:
            # 📌 KEY CONCEPT: Guard Clause (Early Continue)
            # Skip unconfirmed tracks to reduce false positives — cleaner than deep nesting
            if not t.is_confirmed():
                continue

            tid = t.track_id
            active_tids.add(tid)

            # 📌 KEY CONCEPT: dict.get() with Default Value
            # Avoids KeyError on first encounter — returns 0 if track_id is new
            # Count how many frames this track has existed
            self.track_frame_counts[tid] = self.track_frame_counts.get(tid, 0) + 1

            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Get class name safely (DeepSORT can return None)
            name     = (t.get_det_class() or "unknown").lower()
            category = self._classify(name)

            if category == "person":
                persons_tracked[tid]  = {"bbox": (x1, y1, x2, y2), "center": (cx, cy)}
            elif category == "bag":
                bags_tracked[tid]     = {"bbox": (x1, y1, x2, y2), "center": (cx, cy)}
            elif category == "weapon":
                weapon_candidates[tid] = {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "name": name}

        # 📌 KEY CONCEPT: Garbage Collection / Manual Memory Management
        # Remove disappeared track IDs from all state dicts — prevents unbounded memory growth
        # ── Step 2: Clean up memory for tracks that are gone ────────
        for tid in list(self.track_frame_counts.keys()):
            if tid not in active_tids:
                self.track_frame_counts.pop(tid, None)
                self.bag_track_info.pop(tid, None)
                self.weapon_persist.pop(tid, None)
                self.last_weapon_alert_for_tid.pop(tid, None)

        # ── RULE 1: CROWD ────────────────────────────────────────────
        # Only runs if model can detect persons
        if self.has_persons:
            # 📌 KEY CONCEPT: len() on a dict is O(1)
            # No loop needed — Python tracks dict size internally as a counter
            person_count = len(persons_tracked)
            if person_count >= self.crowd_threshold:
                if now - self.last_alert_time["crowd"] >= self.alert_cooldowns["crowd"]:
                    alerts.append({
                        "type": "CROWD",
                        "message": f"Crowd detected: {person_count} people",
                        "timestamp": now,
                        "frame_idx": frame_index
                    })
                    self.last_alert_time["crowd"] = now

        # ── RULE 2: UNATTENDED BAG ───────────────────────────────────
        # Only runs if model can detect bags
        if self.has_bags:
            for tid, info in bags_tracked.items():
                # Wait for track to be stable before alerting
                if self.track_frame_counts.get(tid, 0) < self.min_track_frames_before_bag:
                    continue

                cx, cy = info["center"]

                # First time seeing this bag — record position + time
                if tid not in self.bag_track_info:
                    self.bag_track_info[tid] = {"last_center": (cx, cy), "first_seen": now, "last_seen": now}
                    continue

                entry = self.bag_track_info[tid]

                # 📌 KEY CONCEPT: Euclidean Distance (math.hypot = Pythagoras theorem)
                # math.hypot(dx, dy) = sqrt(dx² + dy²) — measures pixel distance between two centers
                dist  = math.hypot(entry["last_center"][0] - cx, entry["last_center"][1] - cy)

                if dist > 10:
                    # Bag moved — reset the timer
                    entry["last_center"] = (cx, cy)
                    entry["first_seen"]  = now
                entry["last_seen"] = now

                # 📌 KEY CONCEPT: Timestamp-based Stationary Timer (FPS-independent)
                # Using real wall-clock seconds instead of frame counts — works at any FPS
                duration = entry["last_seen"] - entry["first_seen"]

                # 📌 KEY CONCEPT: Generator Expression + any() for Proximity Check
                # Lazy evaluation — any() stops at the first person found within range (no full scan)
                # Check if any person is close to the bag (within 150px)
                has_owner = any(
                    math.hypot(pinfo["center"][0] - cx, pinfo["center"][1] - cy) < 150
                    for pinfo in persons_tracked.values()
                )

                # Alert if bag is alone and has been stationary long enough
                if not has_owner and duration >= self.bag_stationary_seconds:
                    if now - self.last_alert_time["bag"] >= self.alert_cooldowns["bag"]:
                        alerts.append({
                            "type": "UNATTENDED_BAG",
                            "message": f"Bag {tid} unattended for {duration:.1f}s",
                            "timestamp": now, "frame_idx": frame_index,
                            "bbox": info["bbox"], "track_id": tid
                        })
                        self.last_alert_time["bag"] = now
                        entry["first_seen"] = now  # Reset timer after alert

        # ── RULE 3: WEAPON ───────────────────────────────────────────
        # Only runs if model can detect weapons
        if self.has_weapons:
            for tid, winfo in weapon_candidates.items():
                # 📌 KEY CONCEPT: Per-Track Consecutive Frame Counter (Temporal Persistence Filter)
                # A weapon must appear for N consecutive frames before alerting — filters single-frame noise
                # Count consecutive frames this weapon track has been seen
                prev = self.weapon_persist.get(tid, {"count": 0})
                prev["count"] = prev.get("count", 0) + 1
                self.weapon_persist[tid] = prev

                # Alert only after N consecutive frames (reduces false positives)
                if prev["count"] >= self.weapon_persist_frames:
                    # 📌 KEY CONCEPT: Double Cooldown Guard (Global + Per-Track)
                    # Global cooldown: prevents alert flood across all weapons
                    # Per-track cooldown: prevents the same weapon from re-alerting too fast
                    last_tid_time = self.last_weapon_alert_for_tid.get(tid, 0.0)
                    if (now - last_tid_time >= self.alert_cooldowns["weapon"] and
                            now - self.last_alert_time["weapon"] >= self.alert_cooldowns["weapon"]):
                        alerts.append({
                            "type": "WEAPON",
                            "message": f"Weapon ({winfo['name']}) detected!",
                            "timestamp": now, "frame_idx": frame_index,
                            "bbox": winfo["bbox"], "track_id": tid
                        })
                        self.last_alert_time["weapon"] = now
                        self.last_weapon_alert_for_tid[tid] = now

        # Returns: [{"type": "WEAPON", "message": "...", ...}, ...]
        return alerts
