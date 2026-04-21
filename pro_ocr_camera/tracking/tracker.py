from ultralytics import YOLO
import numpy as np

class ObjectTracker:
    def __init__(self, model_path="yolo11n.pt", conf=0.3, tracker="bytetrack.yaml", device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker = tracker
        self.device = device

    def track(self, frame):
        """
        Performs detection and tracking.
        Returns: List of tracked objects with [x1, y1, x2, y2, track_id, conf, cls]
        """
        results = self.model.track(
            source=frame, 
            conf=self.conf, 
            persist=True, 
            tracker=self.tracker, 
            device=self.device, 
            verbose=False
        )
        
        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls in zip(boxes, ids, confs, clss):
                tracks.append((*box, track_id, conf, cls))
                
        return tracks

    def get_crop_by_id(self, frame, track):
        x1, y1, x2, y2, track_id, conf, cls = track
        # Padding to help OCR
        h, w = frame.shape[:2]
        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        return frame[y1:y2, x1:x2], (x1, y1)
