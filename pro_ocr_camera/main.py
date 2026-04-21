import cv2
import time
import threading
from queue import Queue
from config import Config
from camera.camera_stream import CameraStream
from preprocessing.image_utils import ImageUtils
from detection.text_detector import TextDetector
from ocr.ocr_engine import OCRFactory
from utils.draw import DrawingUtils

class OCRApp:
    def __init__(self):
        Config.setup()
        self.stream = CameraStream(
            source=Config.CAMERA_SOURCE,
            width=Config.FRAME_WIDTH,
            height=Config.FRAME_HEIGHT
        ).start()
        
        self.detector = TextDetector(
            model_path=Config.DETECTION_MODEL_PATH,
            conf=Config.DETECTION_CONFIDENCE,
            device=Config.DEVICE
        )
        
        self.ocr_engine = OCRFactory.get_engine(
            engine_type=Config.OCR_ENGINE,
            languages=Config.LANGUAGES,
            gpu=Config.USE_GPU
        )
        
        self.results_queue = Queue(maxsize=1)
        self.latest_results = []
        self.processing_latency = 0
        self.running = True
        
        # Start background processing thread
        self.proc_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.proc_thread.start()

    def process_frames(self):
        """Background thread for heavy heavy computations (Detection + OCR)."""
        frame_count = 0
        while self.running:
            grabbed, frame = self.stream.read()
            if not grabbed or frame is None:
                continue
            
            frame_count += 1
            if frame_count % Config.FRAME_SKIP != 0:
                continue
                
            start_time = time.time()
            
            # 1. Pre-process for detection (optional resize for speed)
            detection_frame = ImageUtils.resize_frame(frame, scale=Config.RESIZE_SCALE)
            
            # 2. Text Detection
            # Note: Since we resized, we need to scale boxes back to original size
            detections = self.detector.detect(detection_frame)
            scale = 1.0 / Config.RESIZE_SCALE
            
            scaled_detections = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                scaled_detections.append((
                    int(x1 * scale), int(y1 * scale), 
                    int(x2 * scale), int(y2 * scale), 
                    conf, cls
                ))
            
            # 3. OCR on detected regions (or full frame if no regions)
            ocr_results = []
            if scaled_detections:
                crops = self.detector.get_cropped_regions(frame, scaled_detections)
                for i, crop in enumerate(crops):
                    # Optional: ImageUtils.prepare_for_ocr(crop)
                    res = self.ocr_engine.recognize(crop)
                    # Adjust OCR bboxes back to global frame coordinates
                    offset_x, offset_y = scaled_detections[i][0], scaled_detections[i][1]
                    for r in res:
                        # This part depends on OCR engine bbox format. 
                        # We'll simplify for the demo and just store text if offset is complex
                        ocr_results.append(r) 
            else:
                # Fallback: process entire frame if no detection models found or specified
                # ocr_results = self.ocr_engine.recognize(frame)
                pass

            self.latest_results = ocr_results
            self.processing_latency = (time.time() - start_time) * 1000
            
    def run(self):
        prev_time = time.time()
        try:
            while True:
                grabbed, frame = self.stream.read()
                if not grabbed or frame is None:
                    break
                
                # Draw latest available results
                frame = DrawingUtils.draw_ocr_results(frame, self.latest_results)
                
                # Calculate display FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time
                
                frame = DrawingUtils.draw_fps(frame, fps, self.processing_latency)
                
                cv2.imshow("Pro Real-Time OCR", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = OCRApp()
    app.run()
