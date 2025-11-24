from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        """
        Update tracker with detections.
        detections: list of (x1, y1, x2, y2, conf, class_name)
        frame: current video frame (numpy array)
        
        Returns: list of deep_sort_realtime.deep_sort.track.Track objects
        """
        # Format for DeepSort: [((x1,y1,x2,y2), conf, class_name), ...]
        ds_input = [((x1, y1, x2, y2), conf, name) for (x1, y1, x2, y2, conf, name) in detections]
        
        tracks = self.tracker.update_tracks(ds_input, frame=frame)
        return tracks
