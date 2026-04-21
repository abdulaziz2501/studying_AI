# Professional Real-time OCR Camera Application

A high-performance OCR system that detects and recognizes text from live camera feeds using YOLOv8 and EasyOCR/Tesseract.

## 🚀 Features

- **Real-time Performance**: Threaded camera capture and asynchronous processing maintain high FPS (15-25+).
- **Advanced Text Detection**: Modular YOLOv8 integration for identifying text regions.
- **Switchable OCR Engines**: Support for EasyOCR (Deep Learning based) and Tesseract.
- **Optimized for GPU**: Automatic CUDA acceleration for detection and OCR.
- **Modular Architecture**: Clean separation of camera, preprocessing, detection, and OCR logic.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   cd pro_ocr_camera
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract (Optional)**:
   If you wish to use the Tesseract engine:
   ```bash
   sudo apt update
   sudo apt install tesseract-ocr
   ```

## 💻 Usage

1. **Configure the application**:
   Edit `config.py` to change settings like:
   - `CAMERA_SOURCE`: Change to RTSP URL or external webcam index.
   - `OCR_ENGINE`: Switch between `"easyocr"` and `"tesseract"`.
   - `RESIZE_SCALE`: Adjust for better performance.

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Interactions**:
   - The UI displays detected text and bounding boxes.
   - Press **'q'** to quit the application.

## 🏗️ Project Structure

- `main.py`: Entry point and main loop.
- `camera/`: Video stream handling.
- `preprocessing/`: Image filtering and resizing.
- `detection/`: YOLOv8 text detection logic.
- `ocr/`: Character recognition engines.
- `utils/`: UI drawing helper functions.
- `config.py`: Global configuration.

## 📊 Performance Tips

- For maximum speed, ensure you have an NVIDIA GPU and CUDA installed.
- Increase `FRAME_SKIP` in `config.py` if the CPU is struggling.
- Use `RESIZE_SCALE = 0.5` or lower to speed up text detection on large frames.
