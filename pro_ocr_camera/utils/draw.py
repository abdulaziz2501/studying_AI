import cv2
import time

class DrawingUtils:
    @staticmethod
    def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"Text Region {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    @staticmethod
    def draw_ocr_results(frame, results, color=(255, 0, 0), font_scale=0.6):
        """
        results: list of dicts with 'text', 'bbox' (can be 4 points or [x,y,w,h])
        """
        for res in results:
            text = res['text']
            bbox = res['bbox']
            
            # EasyOCR returns 4 corners
            if len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
                p1 = tuple(map(int, bbox[0]))
                cv2.rectangle(frame, p1, tuple(map(int, bbox[2])), color, 2)
                cv2.putText(frame, text, (p1[0], p1[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            # Tesseract or simple [x,y,w,h]
            else:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return frame

    @staticmethod
    def draw_fps(frame, fps, latency=None):
        label = f"FPS: {fps:.1f}"
        if latency:
            label += f" | Latency: {latency:.1f}ms"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame
