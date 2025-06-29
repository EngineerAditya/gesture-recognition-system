# I have used ChatGPT to structure and make this code readable 😄
# Run this file to start webcam-based gesture recognition with threaded TTS

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import pyttsx3
import time
import platform
import threading

# ----------------- Load Saved Model & Label Encoder -----------------
print("[INFO] Loading TensorFlow model and label encoder...")

model_path = "models/tf_gesture_model.h5"
le_path = "models/label_encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(le_path):
    print("[ERROR] Trained model or label encoder not found. Please train the model first.")
    exit()

model = tf.keras.models.load_model(model_path)

with open(le_path, "rb") as f:
    le = pickle.load(f)

print("[INFO] Model and Label Encoder loaded successfully!")

# ----------------- Initialize MediaPipe Hands -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ----------------- Initialize Text-to-Speech Engine -----------------
print("[INFO] Initializing TTS engine...")
system_os = platform.system().lower()

if system_os == 'linux':
    engine = pyttsx3.init(driverName='espeak')
else:
    engine = pyttsx3.init()

engine.setProperty('rate', 135)  # Adjust speed
print(f"[INFO] TTS engine ready for {system_os.capitalize()}")

# ----------------- Helper Function for Threaded TTS -----------------
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# ----------------- Start Webcam -----------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam started. Press 'q' to quit.")

# ----------------- Live Prediction Loop -----------------
last_gesture = None
last_spoken_time = 0
cooldown_sec = 3  # Cooldown between speaking

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h1_data = [0] * 63  # Right hand
    h2_data = [0] * 63  # Left hand

    if results.multi_hand_landmarks and results.multi_handedness:
        handedness_info = []
        for hand_idx, handedness in enumerate(results.multi_handedness):
            label = handedness.classification[0].label
            handedness_info.append((label, hand_idx))

        for label, hand_idx in handedness_info:
            landmarks = results.multi_hand_landmarks[hand_idx]
            row = []
            for lm in landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            if label == "Right":
                h1_data = row
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2))
            elif label == "Left":
                h2_data = row
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

    combined_row = h1_data + h2_data

    if any(combined_row):
        X_input = np.array(combined_row).reshape(1, -1)
        prediction_probs = model.predict(X_input, verbose=0)
        predicted_index = np.argmax(prediction_probs, axis=1)[0]
        gesture_name = le.inverse_transform([predicted_index])[0]

        cv.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        current_time = time.time()
        if gesture_name != last_gesture or (current_time - last_spoken_time) >= cooldown_sec:
            threading.Thread(target=speak_text, args=(gesture_name,), daemon=True).start()
            last_spoken_time = current_time
            last_gesture = gesture_name

    cv.imshow("Live Gesture Prediction", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------- Cleanup -----------------
cap.release()
cv.destroyAllWindows()
print("[INFO] Webcam closed.")
