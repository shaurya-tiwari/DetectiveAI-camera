import math

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except Exception:
    DeepSort = None
    HAS_DEEPSORT = False

class Tracker:
    def __init__(self, max_age=30):
        if HAS_DEEPSORT:
            self.tracker = DeepSort(max_age=max_age)
            self.use_deepsort = True
        else:
            self.use_deepsort = False
            self.next_id = 1
            self.objects = {}
            self.max_age = max_age

    def update(self, detections, frame):
        """
        detections: list of (x1,y1,x2,y2,conf,class_name)
        returns list of track-like objects with:
          - track_id
          - is_confirmed()
          - to_ltrb()
          - get_det_class()
        """
        if self.use_deepsort:
            ds_input = [((x1,y1,x2,y2), conf, name) for (x1,y1,x2,y2,conf,name) in detections]
            tracks = self.tracker.update_tracks(ds_input, frame=frame)
            return tracks
        else:
            # simple centroid tracker fallback
            new_objects = {}
            centroids = []
            for (x1,y1,x2,y2,conf,name) in detections:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroids.append(((x1,y1,x2,y2,conf,name), (cx, cy)))
            unmatched = set(self.objects.keys())
            for det, centroid in centroids:
                best_id = None
                best_dist = 1e9
                for oid, obj in self.objects.items():
                    ox, oy = obj["centroid"]
                    dist = math.hypot(ox - centroid[0], oy - centroid[1])
                    if dist < best_dist and dist < 100:
                        best_dist = dist
                        best_id = oid
                if best_id is None:
                    oid = self.next_id
                    self.next_id += 1
                else:
                    oid = best_id
                    if oid in unmatched:
                        unmatched.remove(oid)
                x1,y1,x2,y2,conf,name = det
                cx,cy = centroid
                new_objects[oid] = {"bbox": (int(x1),int(y1),int(x2),int(y2)),
                                    "centroid": (cx,cy),
                                    "name": name,
                                    "age": 0,
                                    "hits": self.objects.get(oid,{}).get("hits",0) + 1}
            for oid in unmatched:
                obj = self.objects.get(oid)
                if obj is not None:
                    obj["age"] = obj.get("age",0) + 1
                    if obj["age"] <= self.max_age:
                        new_objects[oid] = obj
            self.objects = new_objects

            class SimpleTrack:
                def __init__(self, tid, obj):
                    self.track_id = tid
                    self._obj = obj
                def is_confirmed(self):
                    return self._obj.get("hits",0) > 0
                def to_ltrb(self):
                    x1,y1,x2,y2 = self._obj["bbox"]
                    return [x1,y1,x2,y2]
                def get_det_class(self):
                    return self._obj.get("name","unknown")

            return [SimpleTrack(tid,obj) for tid,obj in self.objects.items()]
