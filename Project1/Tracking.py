import cv2
from ultralytics import YOLO

# 1. Model yuklash
model = YOLO("yolov8n.pt")

# 2. Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam ochilmadi!")
    exit()

print("Object Tracking ishga tushdi... (ESC bilan chiqasiz)")

# 3. Tracking uchun persist=True juda muhim!
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. TRACKING (farqi shu!)
    results = model.track(frame, persist=True)

    # 5. Annotatsiya
    annotated_frame = results[0].plot()

    # 6. Ko‘rsatish
    cv2.imshow("Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()