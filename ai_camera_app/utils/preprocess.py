import cv2
import numpy as np

def preprocess_image(frame, target_size=(224, 224)):
    """
    Preprocess image for Keras model prediction.
    - Convert BGR to RGB
    - Resize to target size
    - Normalize pixel values to [0, 1]
    - Expand dimensions for model input (None, H, W, C)
    """
    # BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized_image = cv2.resize(rgb_frame, target_size)
    
    # Normalize
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    # Expand dimensions (batch size 1)
    input_data = np.expand_dims(normalized_image, axis=0)
    
    return input_data

def draw_info(frame, label, confidence, fps):
    """
    Overlay prediction and performance info on the frame.
    """
    text = f"{label} ({confidence:.1%})"
    fps_text = f"FPS: {fps:.1f}"
    
    # Draw label backdrop
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw FPS
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame
