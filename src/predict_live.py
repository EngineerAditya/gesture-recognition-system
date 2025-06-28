# predict_live.py
# Author: Aditya Sinha
# Live gesture prediction using scikit-learn model & MediaPipe
# Directly open this file to start webcam-based gesture prediction

import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
import os

# ---------------- Load Saved Model & Label Encoder ----------------
print("[INFO] Loading model and label encoder...")

model_path = "models/sk_model.pkl"
le_path = "models/label_encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(le_path):
    print("[ERROR] Trained model or label encoder not found. Train the model first.")
    exit()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(le_path, "rb") as f:
    le = pickle.load(f)

print("[INFO] Model & Label Encoder Loaded Successfully!")

# ---------------- Initialize MediaPipe Hands ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------------- Start Webcam ----------------
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Press 'q' to quit window.")

# ---------------- Prediction Loop ----------------
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
            label = handedness.classification[0].label  # 'Left' or 'Right'
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

    # Only predict if at least one hand has valid data (non-zero)
    if any(combined_row):
        X_input = np.array(combined_row).reshape(1, -1)
        prediction = model.predict(X_input)[0]
        gesture_name = le.inverse_transform([prediction])[0]

        cv.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv.imshow("Live Gesture Prediction", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("[INFO] Webcam closed.")
