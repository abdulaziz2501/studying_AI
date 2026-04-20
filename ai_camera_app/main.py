import cv2
import time
import os
import logging
import datetime
import numpy as np

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, 
    DEFAULT_MODEL_PATH, SAVE_DIR, LOG_FILE
)
from camera.camera import WebcamStream
from model.loader import ModelHandler
from model.downloader import download_default_model
from utils.preprocess import preprocess_image, draw_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

def main():
    # 1. Check for model, download if missing
    if not os.path.exists(DEFAULT_MODEL_PATH):
        logger.info("Default model path not found.")
        download_default_model(DEFAULT_MODEL_PATH)

    # 2. Initialize Model
    model_handler = ModelHandler(DEFAULT_MODEL_PATH)
    if not model_handler.load_model():
        logger.error("Could not load model. Exiting.")
        return

    # 3. Initialize Camera
    try:
        stream = WebcamStream(src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
        logger.info("Camera stream started.")
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        return

    # 4. Prepare directories
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 5. Main Loop
    prev_time = time.time()
    fps = 0
    
    # Pre-loading Label decoder if using ImageNet
    decode_predictions = model_handler.get_labels()

    logger.info("Starting real-time detection. Press 'q' to quit, 's' to save frame.")
    
    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # Performance Timing
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # Preprocessing
            input_data = preprocess_image(frame, target_size=model_handler.input_shape)

            # Prediction
            preds = model_handler.predict(input_data)
            
            # Post-processing (Top-1 classification for this demo)
            label = "Unknown"
            confidence = 0.0
            
            if preds is not None:
                if decode_predictions:
                    # Results format: [[(class_id, class_name, confidence), ...]]
                    results = decode_predictions(preds, top=1)[0]
                    _, label, confidence = results[0]
                else:
                    # Fallback for generic models: just show argmax or similar
                    label = f"Class {np.argmax(preds[0])}"
                    confidence = float(np.max(preds[0]))

            # Visualization
            display_frame = draw_info(frame, label, confidence, fps)
            
            # Show Feed
            cv2.imshow("AI Real-time Detection", display_frame)

            # Interaction
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit command received.")
                break
            elif key == ord('s'):
                filename = os.path.join(SAVE_DIR, f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(filename, frame)
                logger.info(f"Frame saved to {filename}")

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        logger.info("Cleaning up...")
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
