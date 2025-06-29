# I have used ChatGPT to structure and make this code readable 😄
# Run this file to start webcam-based gesture recognition with threaded speech and stable CV text

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

engine.setProperty('rate', 135)
print(f"[INFO] TTS engine ready for {system_os.capitalize()}")

speech_lock = threading.Lock()
last_spoken = None
last_spoken_time = 0
cooldown_sec = 3
last_prediction_time = 0
prediction_delay = 1  # 1 second delay between predictions
displayed_gesture = "No Gesture"

def speak(gesture):
    global last_spoken, last_spoken_time
    with speech_lock:
        current_time = time.time()
        if gesture != last_spoken or (current_time - last_spoken_time) > cooldown_sec:
            engine.say(gesture)
            engine.runAndWait()
            last_spoken = gesture
            last_spoken_time = current_time

# ----------------- Start Webcam -----------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam started. Press 'q' to quit.")

# ----------------- Live Prediction Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h1_data = [0] * 63
    h2_data = [0] * 63

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
    current_time = time.time()

    if any(combined_row) and (current_time - last_prediction_time) > prediction_delay:
        X_input = np.array(combined_row).reshape(1, -1)
        prediction_probs = model.predict(X_input, verbose=0)
        predicted_index = np.argmax(prediction_probs, axis=1)[0]
        gesture_name = le.inverse_transform([predicted_index])[0]

        displayed_gesture = gesture_name
        threading.Thread(target=speak, args=(gesture_name,), daemon=True).start()

        last_prediction_time = current_time

    # Always show the last predicted gesture on screen
    cv.putText(frame, f"Gesture: {displayed_gesture}", (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Live Gesture Prediction", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------- Cleanup -----------------
cap.release()
cv.destroyAllWindows()
print("[INFO] Webcam closed.")
