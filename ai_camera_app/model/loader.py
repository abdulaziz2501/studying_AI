import os
import logging
import tensorflow as tf
from config import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224)
        
    def load_model(self):
        """Loads the Keras model and detects input shape."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            logger.info(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Detect input shape from model metadata
            # Typical shape is (None, H, W, C)
            try:
                input_layer = self.model.layers[0]
                # Some models have InputLayer as first layer, others might have something else
                config = input_layer.get_config()
                if 'batch_input_shape' in config:
                    shape = config['batch_input_shape'] # (None, 224, 224, 3)
                else:
                    shape = self.model.input_shape # (None, 224, 224, 3)
                
                self.input_shape = (shape[1], shape[2])
                logger.info(f"Auto-detected input shape: {self.input_shape}")
            except Exception as e:
                logger.warning(f"Could not auto-detect shape, using default (224, 224). Error: {e}")
                self.input_shape = (224, 224)
                
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, preprocessed_frame):
        """Performs inference on the frame."""
        if self.model is None:
            return None
        
        predictions = self.model.predict(preprocessed_frame, verbose=0)
        return predictions

    def get_labels(self):
        """Helper to get ImageNet labels if using default MobileNetV2."""
        # This is a simplification. For a real production app, 
        # labels should be loaded from a separate file.
        try:
            from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
            return decode_predictions
        except ImportError:
            return None
