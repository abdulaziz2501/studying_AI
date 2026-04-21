import os
import torch

class Config:
    # --- Camera Settings ---
    CAMERA_SOURCE = 0  # 0 for default webcam, or path/URL to video
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    
    # --- Performance Optimization ---
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if USE_GPU else "cpu"
    FRAME_SKIP = 2  # Process detection every N frames
    RESIZE_SCALE = 0.5  # Scale factor for inference to boost speed
    
    # --- Detection & Tracking ---
    DETECTION_MODEL_PATH = "yolo11n.pt"  # Upgraded to yolo11 for better performance
    DETECTION_CONFIDENCE = 0.3
    TRACKER_TYPE = "bytetrack.yaml"  # Using YOLOv8/v11 native ByteTrack
    
    # --- OCR Settings ---
    OCR_ENGINE = "easyocr"  # "easyocr" or "tesseract"
    LANGUAGES = ["en"]      
    OCR_CONFIDENCE_THRESHOLD = 0.6
    
    # --- Specialized Use-case: LPR (License Plate Recognition) ---
    LPR_REGEX = r'^[A-Z0-9-]{3,8}$'  # Generic alphanumeric plate regex
    CHAR_CORRECTIONS = {
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '8'
    }
    SMOOTHING_WINDOW = 10  # Number of frames to keep history for voting
    
    # --- API Settings ---
    API_HOST = "0.0.0.0"
    API_PORT = 8005
    
    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    @classmethod
    def setup(cls):
        for d in [cls.LOG_DIR, cls.OUTPUT_DIR]:
            if not os.path.exists(d):
                os.makedirs(d)

if __name__ == "__main__":
    Config.setup()
