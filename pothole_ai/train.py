import os
import yaml
from ultralytics import YOLO

# Set working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Initialize model
    model = YOLO(cfg['model']['model_type'])

    # Training parameters
    train_params = {
        "data": cfg['training']['data'],
        "epochs": cfg['training']['epochs'],
        "imgsz": cfg['model']['img_size'],
        "batch": cfg['training']['batch_size'],
        "device": cfg['training']['device'],
        "workers": cfg['training']['workers'],
        "optimizer": cfg['training']['optimizer'],
        "lr0": cfg['training']['lr0'],
        "augment": True,
        "half": True,       # Use FP16 for training speedup
        "val": True,
        "plots": True,
        "name": "pothole_segmentation_v8"
    }

    print(f"Starting training for {train_params['epochs']} epochs on device {train_params['device']}...")
    
    # Run training
    results = model.train(**train_params)
    
    print("Training complete. Models saved in 'runs/segment/pothole_segmentation_v8/weights/'")
    
    # Export to FP16 for optimized inference
    print("Exporting optimized model...")
    model.export(format="engine", half=True) # Exporting to TensorRT (engine) if possible, otherwise keep .pt

if __name__ == "__main__":
    train_model()
