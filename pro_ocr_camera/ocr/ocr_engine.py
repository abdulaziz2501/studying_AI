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
        # Initialize reader once
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def recognize(self, image):
        """Returns: List of detected text strings."""
        if image is None or image.size == 0:
            return []
        try:
            results = self.reader.readtext(image)
            return [{"text": res[1], "confidence": res[2], "bbox": res[0]} for res in results]
        except Exception as e:
            print(f"OCR Error: {e}")
            return []

class TesseractEngine(BaseOCREngine):
    def __init__(self):
        pass

    def recognize(self, image):
        if image is None or image.size == 0:
            return []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            results = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    text = data['text'][i].strip()
                    if text:
                        results.append({
                            "text": text,
                            "confidence": data['conf'][i] / 100.0,
                            "bbox": [data['left'][i], data['top'][i], data['width'][i], data['height'][i]]
                        })
            return results
        except Exception as e:
            print(f"Tesseract Error: {e}")
            return []

class OCRFactory:
    _engines = {}

    @classmethod
    def get_engine(cls, engine_type="easyocr", languages=['en'], gpu=True):
        key = f"{engine_type}_{''.join(languages)}_{gpu}"
        if key not in cls._engines:
            if engine_type == "easyocr":
                cls._engines[key] = EasyOCREngine(languages, gpu)
            elif engine_type == "tesseract":
                cls._engines[key] = TesseractEngine()
            else:
                raise ValueError(f"Unknown OCR engine: {engine_type}")
        return cls._engines[key]
