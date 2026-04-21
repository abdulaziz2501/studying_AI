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
    FRAME_SKIP = 2  # Process text detection/OCR every N frames
    RESIZE_SCALE = 0.5  # Scale factor for detection frame (smaller = faster)
    
    # --- Detection Settings (YOLOv3/v8/EAST) ---
    # Using YOLOv8n as default. For specific text detection, 
    # paths to custom weights can be placed here.
    DETECTION_MODEL_PATH = "yolo26l.pt"  
    DETECTION_CONFIDENCE = 0.25
    
    # --- OCR Settings ---
    OCR_ENGINE = "easyocr"  # "easyocr" or "tesseract"
    LANGUAGES = ["en"]      # List of languages for OCR
    
    # Filtering settings
    ONLY_NUMBERS = False
    UPPERCASE_ONLY = False
    
    # --- Visualization Settings ---
    SHOW_FPS = True
    BOX_COLOR = (0, 255, 0)  # BGR
    TEXT_COLOR = (255, 255, 255)
    FONT_SCALE = 0.6
    THICKNESS = 2

    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    
    @classmethod
    def setup(cls):
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)

if __name__ == "__main__":
    Config.setup()
    print(f"Config initialized. Device: {Config.DEVICE}")
