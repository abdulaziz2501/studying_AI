import cv2
import torch
import time
import yaml
import os
from ultralytics import YOLO

# Set working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import filter_by_area, draw_visuals, display_info

def run_inference():
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Model (Optimized for GPU)
    model_path = cfg['model']['model_type']
    device = f"cuda:{cfg['training']['device']}" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_path} on {device}...")
    model = YOLO(model_path)
    
    # RTX 3080 Optimization: Move to GPU and set to FP16
    model.to(device)
    if cfg['inference']['fp16'] and device != "cpu":
        model.model.half()
        print("Using FP16 (Half Precision) for inference acceleration.")

    # 2. Model Warm-up
    print("Warming up the engine...")
    dummy_input = torch.zeros(1, 3, cfg['model']['img_size'], cfg['model']['img_size']).to(device)
    if cfg['inference']['fp16'] and device != "cpu":
        dummy_input = dummy_input.half()
    for _ in range(10):
        _ = model(dummy_input, verbose=False)

    # 3. Setup Video Stream
    source = cfg['inference']['source']
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Starting real-time pothole detection. Press 'q' to exit.")

    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished video or lost connection.")
            break

        # 4. Run YOLOv8 Segmentation
        results = model.predict(
            source=frame,
            conf=cfg['inference']['conf_threshold'],
            iou=cfg['inference']['iou_threshold'],
            imgsz=cfg['model']['img_size'],
            half=cfg['inference']['fp16'],
            device=device,
            verbose=False
        )

        # 5. Process Results
        # Filter by area to remove noise
        filtered_indices = filter_by_area(results, min_area=cfg['inference']['min_mask_area'])
        
        # Draw Masks, Boxes and Counter
        frame, count = draw_visuals(frame, results, filtered_indices)

        # 6. Calculate and Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        frame = display_info(frame, count, fps)

        # 7. Show Frame
        cv2.imshow("RTX 3080 Pothole AI System", frame)

        # Handle 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    run_inference()
