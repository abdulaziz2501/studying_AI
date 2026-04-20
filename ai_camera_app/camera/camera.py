import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class WebcamStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.stream.isOpened():
            logger.error(f"Cannot open camera index {src}")
            raise RuntimeError(f"Could not open camera {src}")

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        """Start the thread to read frames from the video stream."""
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        """Keep looping infinitely until the thread is stopped."""
        while True:
            if self.stopped:
                return

            (grabbed, frame) = self.stream.read()
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
                
            if not grabbed:
                logger.warning("Camera disconnected or failed to read frame.")
                self.stop()
                return

    def read(self):
        """Return the frame most recently read."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True
        if self.stream:
            self.stream.release()
