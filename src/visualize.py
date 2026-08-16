import cv2
import time

# ┌──────────────────────────────────────────────────────────────────┐
# │                    VISUALIZE MODULE                              │
# │  Draws bounding boxes + alert text on each video frame           │
# │                                                                  │
# │  draw_tracks() → draws a box + ID label for every tracked object │
# │  draw_alerts() → draws alert text at top of frame in red         │
# └──────────────────────────────────────────────────────────────────┘


# ┌──────────────────────────────────────────────────────────────────┐
# │  DRAW TRACKS                                                     │
# │  For every tracked object → draw a yellow bounding box           │
# │  and show the track ID + class name above it                     │
# │                                                                  │
# │  Input:  frame + list of tracks from tracker                     │
# │  Output: frame with boxes drawn on it                            │
# └──────────────────────────────────────────────────────────────────┘
def draw_tracks(frame, tracks):
    # 📌 KEY CONCEPT: Defensive Copy (frame.copy())
    # We draw on a copy, not the original frame — prevents side-effects if the frame is reused elsewhere
    out = frame.copy()

    for t in tracks:
        try:
            # Get [x1, y1, x2, y2] bounding box
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]

            # Get track ID and class name
            tid  = getattr(t, "track_id", None)
            name = t.get_det_class() if hasattr(t, "get_det_class") else "obj"

            # 📌 KEY CONCEPT: OpenCV Drawing API (Mutable In-Place Operations)
            # cv2.rectangle and cv2.putText modify the numpy array in place — very fast, no new allocations
            # Draw the box (yellow colour)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)

            # Draw label above the box
            label = f"ID:{tid} {name}"
            cv2.putText(out, label, (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        except Exception:
            pass  # Skip this track if any value is missing/invalid

    return out


# ┌──────────────────────────────────────────────────────────────────┐
# │  DRAW ALERTS                                                     │
# │  Prints up to 5 alert messages at the top-left of the frame      │
# │  in red text so they're visible on any background                │
# │                                                                  │
# │  Input:  frame + list of current alert dicts                     │
# │  Output: frame with alert text overlaid                          │
# └──────────────────────────────────────────────────────────────────┘
def draw_alerts(frame, alerts):
    # 📌 KEY CONCEPT: Defensive Copy + List Slicing [:5]
    # Drawing on a copy avoids mutating the original frame; [:5] caps output to prevent UI overflow
    out = frame.copy()
    y = 20  # Starting Y position for first line of text

    for a in alerts[:5]:  # Show max 5 alerts on-screen at once
        # 📌 KEY CONCEPT: Unix Timestamp → Human-Readable Time (time.strftime)
        # Alert timestamps are stored as float seconds (Unix epoch) — strftime formats them for display
        # Format: "20:13:07 [WEAPON] Weapon (pistol) detected!"
        ts  = time.strftime('%H:%M:%S', time.localtime(a['timestamp']))
        txt = f"{ts} [{a['type']}] {a['message']}"

        # Draw red text onto the frame
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 25  # Move down for next alert line

    return out
