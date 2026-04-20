import os

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_UPDATE_INTERVAL = 1.0  # seconds

# Model Settings
MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")
INPUT_SHAPE = (224, 224, 3)  # default, will be overwritten by loader if different

# Feature Settings
SAVE_DIR = "captured_frames"
LOG_FILE = "app.log"

# UI Settings
FONT_SCALE = 0.7
FONT_THICKNESS = 2
COLOR_READY = (0, 255, 0)
COLOR_ERROR = (0, 0, 255)
