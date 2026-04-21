import cv2
import numpy as np

class ImageUtils:
    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def reduce_noise(image):
        # Apply Gaussian Blur or Bilateral Filter
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    @staticmethod
    def apply_threshold(image):
        # Adaptive thresholding or Otsu's
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def prepare_for_ocr(image):
        """Standard preprocessing pipeline for OCR improvement."""
        gray = ImageUtils.to_grayscale(image)
        # We don't always want thresholding for EasyOCR as it handles color well,
        # but Tesseract often performs better with binary images.
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = ImageUtils.apply_threshold(denoised)
        return thresh

    @staticmethod
    def resize_frame(image, scale=0.5):
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
