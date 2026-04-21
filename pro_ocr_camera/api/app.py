import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from PIL import Image
from config import Config
from tracking.tracker import ObjectTracker
from ocr.ocr_engine import OCRFactory
from postprocessing.text_cleaner import TextCleaner

app = FastAPI(title="Pro OCR API", description="Production-ready OCR / LPR API")

# Initialize global engines
tracker = ObjectTracker(Config.DETECTION_MODEL_PATH, device=Config.DEVICE)
ocr_engine = OCRFactory.get_engine(Config.OCR_ENGINE, gpu=Config.USE_GPU)
cleaner = TextCleaner(Config.LPR_REGEX, Config.CHAR_CORRECTIONS)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image"}

    # Execute tracking (for single image, it's just detection)
    tracks = tracker.track(image)
    
    results = []
    for track in tracks:
        crop, (x1, y1) = tracker.get_crop_by_id(image, track)
        ocr_res = ocr_engine.recognize(crop)
        
        # Take the best OCR result for this crop
        raw_text = ocr_res[0]['text'] if ocr_res else ""
        clean_text = cleaner.clean(raw_text)
        
        results.append({
            "track_id": int(track[4]),
            "confidence": float(track[5]),
            "bbox": [int(track[0]), int(track[1]), int(track[2]), int(track[3])],
            "text": clean_text,
            "raw_text": raw_text
        })
        
    return {"detections": results}

@app.get("/health")
def health():
    return {"status": "ok", "device": Config.DEVICE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
