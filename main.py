import mediapipe as mp
import cv2
import numpy as np
import pyttsx3
from numba import jit
import threading

# Text-to-speech function
def speak_count(counter):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(f"{counter}")
    engine.runAndWait()

# Angle calculation with Numba for optimization
@jit(nopython=True)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Buffer for smoothing angles
angle_buffer = []

def smooth_angle(angle, buffer_size=5):
    angle_buffer.append(angle)
    if len(angle_buffer) > buffer_size:
        angle_buffer.pop(0)
    return sum(angle_buffer) / len(angle_buffer)

# Dumbbell curl detection function
def dumbbell_curl(landmarks, mp_pose, counter, stage):
    try:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)
        angle = smooth_angle(angle)

        if angle > 160 and stage != "down":
            stage = "down"
        if angle < 30 and stage == "down":
            stage = "up"
            counter += 1
            threading.Thread(target=speak_count, args=(counter,)).start()
    except Exception as e:
        print(f"Error in dumbbell_curl: {e}")

    return counter, stage


# Squat detection function
def squats(landmarks, mp_pose, counter, stage):
    try:
        
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

       
        angle = calculate_angle(hip, knee, ankle)

       
        angle = smooth_angle(angle)

       
        if angle > 160 and stage != "up":
            stage = "up"
        elif angle < 100 and stage == "up":  
            stage = "down"
            counter += 1
            threading.Thread(target=speak_count, args=(counter,)).start()
    except Exception as e:
        print(f"Error in squats: {e}")

    return counter, stage

# Main application
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    exercise = None
    instructions = "Press 1 for Curls | Press 2 for Squats | Press Q to Quit"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video feed. Check your webcam.")
                break

            cv2.putText(frame, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            if exercise:
                cv2.putText(frame, f"Exercise: {exercise}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                if exercise == "Curls":
                    counter, stage = dumbbell_curl(results.pose_landmarks.landmark, mp_pose, counter, stage)
                elif exercise == "Squats":
                    counter, stage = squats(results.pose_landmarks.landmark, mp_pose, counter, stage)

            cv2.putText(frame, f"Reps: {counter}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Fitness AI", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                exercise = "Curls"
                counter, stage = 0, None
            elif key == ord('2'):
                exercise = "Squats"
                counter, stage = 0, None
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
