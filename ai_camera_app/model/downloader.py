import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_default_model(save_path="models/model.h5"):
    """Downloads MobileNetV2 as a default model for demonstration."""
    if os.path.exists(save_path):
        logger.info(f"Model already exists at {save_path}")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    logger.info("No model found. Downloading MobileNetV2 as a default...")
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    download_default_model()
