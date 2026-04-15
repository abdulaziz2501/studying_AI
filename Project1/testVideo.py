import cv2

cap = cv2.VideoCapture("/dev/video2")  # OBS camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("OBS Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()