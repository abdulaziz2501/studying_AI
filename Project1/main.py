import cv2
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 640, 480)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("AI Object Detection is run... (exit is ESC)")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (1080, 720))
    results=model(frame)
    annotated_frame= results[0].plot()

    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()