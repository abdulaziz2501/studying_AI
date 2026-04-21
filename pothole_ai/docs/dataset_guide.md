# 🛣️ Pothole Detection Dataset Guide

This guide explains how to prepare your dataset for the Pothole Detection system using YOLOv8 Segmentation.

## 1. Data Collection / Ma'lumotlarni yig'ish
- **Sources**: Use dashcam footage, drone imagery, or public datasets (like RDD2022).
- **Diversity**: Include images from different weather conditions (rainy, sunny) and lighting (night, day).
- **Uzbek**: Dashcam videolari yoki ochiq datasetlardan foydalaning. Turli ob-havo va yorug'lik holatlarini inobatga oling.

## 2. Annotation / Annotatsiya qilish
We recommend using **Roboflow** for annotation.

### Steps:
1. **Upload** your images to Roboflow.
2. **Select "Instance Segmentation"** as the project type.
3. **Tool**: Use the **Polygon Tool** (P) to trace the exact shape of each pothole. 
    - *Crucial*: Do not use bounding boxes; draw the mask along the edge of the hole.
4. **Class Name**: Use `pothole`.

### O'zbekcha yo'riqnoma:
1. Roboflow-ga rasmlarni yuklang.
2. Loyiha turi sifatida **"Instance Segmentation"** tanlang.
3. **Polygon Tool** yordamida chuqurning shaklini aniq chizib chiqing.
4. Klass nomini `pothole` deb yozing.

## 3. Export / Eksport qilish
1. Go to **Export Dataset**.
2. Select **YOLOv8** format.
3. Download the zip and extract it into the `pothole_ai/data/` folder.

The structure should look like this:
```
pothole_ai/data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 4. `data.yaml` Setup
Update your `pothole_ai/data/data.yaml` with the correct paths to your images.

---

# 🚀 Optimization Steps for RTX 3080

1. **FP16 Inference**: Enabled by default in `detect.py`. It uses Tensor Cores to double the performance.
2. **Model Choice**: 
   - `yolov8n-seg.pt` for maximum speed (>=100 FPS).
   - `yolov8s-seg.pt` for better accuracy with slight speed cost.
3. **TensorRT (Optional)**: For production, export the model using `model.export(format='engine')` to get the lowest latency possible.
