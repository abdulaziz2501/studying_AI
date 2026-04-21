import easyocr
import pytesseract
import numpy as np
from abc import ABC, abstractmethod

class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, image):
        pass

class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages=['en'], gpu=True):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def recognize(self, image):
        """Returns: List of detected text strings."""
        # EasyOCR returns (bbox, text, prob)
        results = self.reader.readtext(image)
        return [{"text": res[1], "confidence": res[2], "bbox": res[0]} for res in results]

class TesseractEngine(BaseOCREngine):
    def __init__(self):
        # Tesseract usually requires installation: sudo apt install tesseract-ocr
        pass

    def recognize(self, image):
        # image should be preprocessed for better Tesseract results
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Filter out empty or low conf
                text = data['text'][i].strip()
                if text:
                    results.append({
                        "text": text,
                        "confidence": data['conf'][i] / 100.0,
                        "bbox": [data['left'][i], data['top'][i], data['width'][i], data['height'][i]]
                    })
        return results

class OCRFactory:
    @staticmethod
    def get_engine(engine_type="easyocr", languages=['en'], gpu=True):
        if engine_type == "easyocr":
            return EasyOCREngine(languages, gpu)
        elif engine_type == "tesseract":
            return TesseractEngine()
        else:
            raise ValueError(f"Unknown OCR engine: {engine_type}")
