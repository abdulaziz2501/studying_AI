import cv2
from fer import FER

# Camera
cap = cv2.VideoCapture(2)

# Emotion detector
detector = FER()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Emotion detection
    result = detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        # Eng katta emotion
        emotion = max(emotions, key=emotions.get)
        score = emotions[emotion]

        # Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Text
        cv2.putText(frame, f"{emotion}: {score:.2f}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()