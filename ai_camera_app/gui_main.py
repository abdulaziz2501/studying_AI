import cv2
import time
import os
import logging
import numpy as np
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_MODEL_PATH
from camera.camera import WebcamStream
from model.loader import ModelHandler
from model.downloader import download_default_model
from utils.preprocess import preprocess_image, draw_info
from gui.app_window import CameraGUI

def run_detection_logic(gui_callback):
    """Headless logic that sends frames to the GUI callback."""
    # Initialization
    if not os.path.exists(DEFAULT_MODEL_PATH):
        download_default_model(DEFAULT_MODEL_PATH)

    model_handler = ModelHandler(DEFAULT_MODEL_PATH)
    if not model_handler.load_model():
        return

    try:
        stream = WebcamStream(src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
    except:
        return

    prev_time = time.time()
    decode_predictions = model_handler.get_labels()

    try:
        while True:
            frame = stream.read()
            if frame is None: continue

            # Logic
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            input_data = preprocess_image(frame, target_size=model_handler.input_shape)
            preds = model_handler.predict(input_data)
            
            label = "Unknown"
            confidence = 0.0
            if preds is not None:
                if decode_predictions:
                    results = decode_predictions(preds, top=1)[0]
                    _, label, confidence = results[0]
                else:
                    label = f"Class {np.argmax(preds[0])}"
                    confidence = float(np.max(preds[0]))

            display_frame = draw_info(frame, label, confidence, fps)
            
            # Send to GUI
            gui_callback(display_frame)
            
            # Slow down slightly to not overwhelm GUI thread
            time.sleep(0.01)

    except Exception as e:
        print(f"Logic Error: {e}")
    finally:
        stream.stop()

if __name__ == "__main__":
    gui = CameraGUI(run_detection_logic)
    gui.run()
