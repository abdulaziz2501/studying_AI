# Production-Ready Real-time OCR & LPR System

A high-performance, scalable, and robust License Plate Recognition (LPR) system designed for real-world production environments.

## 🌟 Pro Features

- **3-Thread Architecture**: Decoupled Camera, Inference (Tracking), and OCR pipelines for maximum throughput.
- **Deep Tracking**: Powered by YOLOv8 ByteTrack/BoT-SORT to maintain object continuity.
- **Temporal Smoothing**: Majority-voting system to stabilize OCR results and prevent flickering.
- **LPR Specialization**: Domain-specific regex cleaning and character correction (e.g., misreads like O/0).
- **Production API**: Built-in FastAPI REST service for remote inference.
- **Dockerized**: Ready for containerized deployment in minutes.

## 🏗️ Architecture

1.  **Thread 1 (Main/UI)**: Captures video and renders the stabilized overlays at high FPS.
2.  **Thread 2 (Inference)**: Runs YOLOv8 tracking to identify and follow plates.
3.  **Thread 3 (OCR)**: Targeted recognition on specific track IDs. Results are cached and smoothed over time.

## 🛠️ Setup & Installation

### 1. Clone & Install
```bash
git clone <repo_url>
cd pro_ocr_camera
pip install -r requirements.txt
```

### 2. Run with Docker (Recommended)
```bash
docker build -t pro-ocr-system .
docker run -p 8000:8000 pro-ocr-system
```

### 3. Run Locally
- **Real-time Camera**: `python main.py`
- **REST API Server**: `python -m uvicorn api.app:app --reload`

## ⚙️ Configuration
Modify `config.py` to tune the system:
- `LPR_REGEX`: Customize to your region's plate format.
- `SMOOTHING_WINDOW`: Adjust for more or less "memory" in text stabilization.
- `FRAME_SKIP`: Optimize detection load on legacy CPUs.

## 📊 Performance Metrics (Example)
- **Preview FPS**: 60+ (independent of OCR)
- **Tracking Latency**: ~15ms (YOLO11n-based)
- **OCR Latency**: ~100-200ms (Asynchronized)
- **Stability**: OCR results stabilize within 3-5 frames of detection.

## 📸 Usage
- Press **'q'** to quit.
- Press **'s'** to save a timestamped screenshot of current detections.
