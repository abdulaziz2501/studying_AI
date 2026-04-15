import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

counter = 0
stage = None
score = 100

def angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    ang = abs(radians * 180.0 / np.pi)
    if ang > 180:
        ang = 360 - ang
    return ang

while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        lm = results.pose_landmarks.landmark

        # 🦵 key points
        hip = [lm[23].x, lm[23].y]
        knee = [lm[25].x, lm[25].y]
        ankle = [lm[27].x, lm[27].y]

        shoulder = [lm[11].x, lm[11].y]

        knee_angle = angle(hip, knee, ankle)
        hip_angle = angle(shoulder, hip, knee)

        # =========================
        # 🔥 FORM CHECK LOGIC
        # =========================

        score = 100

        # ❌ Error 1: not deep enough squat
        if knee_angle > 120:
            score -= 30

        # ❌ Error 2: too deep / unsafe knee
        if knee_angle < 70:
            score -= 20

        # ❌ Error 3: bad posture (leaning too much)
        if hip_angle < 40:
            score -= 25

        # =========================
        # 🏋️ COUNT LOGIC
        # =========================

        if knee_angle < 90:
            stage = "down"

        if knee_angle > 160 and stage == "down":
            stage = "up"
            counter += 1

        # =========================
        # 🎯 DISPLAY UI
        # =========================

        if score > 80:
            feedback = "Good Form 💪"
            color = (0, 255, 0)
        elif score > 50:
            feedback = "Average ⚠️"
            color = (0, 255, 255)
        else:
            feedback = "Bad Form ❌"
            color = (0, 0, 255)

        cv2.putText(image, f"Squats: {counter}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(image, f"Score: {score}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(image, feedback, (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(image, f"Knee: {int(knee_angle)}", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    except:
        pass

    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("AI Fitness Trainer PRO", image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()