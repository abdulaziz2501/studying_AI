import cv2
import time
import threading
from queue import Queue
from config import Config
from camera.camera_stream import CameraStream
from tracking.tracker import ObjectTracker
from ocr.ocr_engine import OCRFactory
from postprocessing.text_cleaner import TextCleaner
from postprocessing.smoother import TemporalSmoother
from utils.draw import DrawingUtils

class ProductionOCRSystem:
    def __init__(self):
        Config.setup()
        # Queues
        self.inference_queue = Queue(maxsize=2)
        self.ocr_queue = Queue(maxsize=10)
        
        # Engines
        self.stream = CameraStream(Config.CAMERA_SOURCE).start()
        self.tracker = ObjectTracker(Config.DETECTION_MODEL_PATH, device=Config.DEVICE)
        self.ocr_engine = OCRFactory.get_engine(Config.OCR_ENGINE, gpu=Config.USE_GPU)
        self.cleaner = TextCleaner(Config.LPR_REGEX, Config.CHAR_CORRECTIONS)
        self.smoother = TemporalSmoother(Config.SMOOTHING_WINDOW)
        
        # State
        self.running = True
        self.lock = threading.Lock()
        self.active_tracks = {} # {track_id: {"bbox": [], "text": "", "conf": 0}}
        self.fps_avg = 0
        self.latency_inference = 0
        self.latency_ocr = 0

    def inference_loop(self):
        """Thread 2: Tracking"""
        while self.running:
            grabbed, frame = self.stream.read()
            if not grabbed or frame is None:
                continue
            
            start_t = time.time()
            tracks = self.tracker.track(frame)
            self.latency_inference = (time.time() - start_t) * 1000
            
            current_ids = []
            with self.lock:
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf, cls = track
                    current_ids.append(track_id)
                    
                    # Update bbox immediately for UI
                    if track_id not in self.active_tracks:
                        self.active_tracks[track_id] = {"bbox": [x1, y1, x2, y2], "text": "Detecting...", "conf": conf}
                    else:
                        self.active_tracks[track_id]["bbox"] = [x1, y1, x2, y2]
                    
                    # Push to OCR queue if it's a new track or we want to re-verify
                    if not self.ocr_queue.full():
                        crop, offset = self.tracker.get_crop_by_id(frame, track)
                        self.ocr_queue.put((track_id, crop))

                # Cleanup smoother and active tracks
                self.smoother.cleanup_old_tracks(current_ids)
                dead_ids = [tid for tid in self.active_tracks if tid not in current_ids]
                for tid in dead_ids:
                    del self.active_tracks[tid]
            
            time.sleep(0.01)

    def ocr_loop(self):
        """Thread 3: OCR"""
        while self.running:
            try:
                track_id, crop = self.ocr_queue.get(timeout=1)
            except:
                continue
                
            start_t = time.time()
            ocr_results = self.ocr_engine.recognize(crop)
            self.latency_ocr = (time.time() - start_t) * 1000
            
            if ocr_results:
                raw_text = ocr_results[0]['text']
                clean_text = self.cleaner.clean(raw_text)
                stable_text = self.smoother.update(track_id, clean_text)
                
                with self.lock:
                    if track_id in self.active_tracks:
                        self.active_tracks[track_id]["text"] = stable_text
            
            self.ocr_queue.task_done()

    def run(self):
        # Start background threads
        t_inf = threading.Thread(target=self.inference_loop, daemon=True)
        t_ocr = threading.Thread(target=self.ocr_loop, daemon=True)
        t_inf.start()
        t_ocr.start()
        
        prev_time = time.time()
        
        try:
            while True:
                grabbed, frame = self.stream.read()
                if not grabbed or frame is None:
                    break
                
                # Render UI
                with self.lock:
                    items = list(self.active_tracks.items())
                    
                for tid, data in items:
                    x1, y1, x2, y2 = data["bbox"]
                    text = data["text"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid} {text}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Global Stats
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time
                
                stats = f"FPS: {fps:.1f} | Inf: {self.latency_inference:.1f}ms | OCR: {self.latency_ocr:.1f}ms"
                cv2.putText(frame, stats, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.imshow("Production Real-Time LPR", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'): # Screenshot feature
                    fname = f"outputs/cap_{int(time.time())}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"Saved screenshot: {fname}")
                    
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = ProductionOCRSystem()
    system.run()
