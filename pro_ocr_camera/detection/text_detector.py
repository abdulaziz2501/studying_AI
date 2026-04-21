from ultralytics import YOLO
import cv2
import numpy as np

class TextDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.25, device="cpu"):
        """
        Initialize the YOLOv8 detector.
        Note: For standard OCR, a model trained on text detection (e.g. YOLOv8n-text)
        is preferred over COCO models.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        
    def detect(self, frame):
        """
        Detect potential text regions.
        Returns: List of (x1, y1, x2, y2, confidence, class_id)
        """
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = box.cls[0]
                detections.append((*r, conf, cls))
                
        return detections

    def get_cropped_regions(self, frame, detections):
        """
        Extract cropped images from the frame based on detections.
        """
        crops = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops
