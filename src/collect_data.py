# I have used ChatGPT to structure and make this code readable 😄
# Purpose: Record gesture video + landmark data with MediaPipe and save to CSV
# Notes: Tkinter input for gesture name, recording confirmation popup included

import cv2 as cv
import mediapipe as mp
import pandas as pd
import os
import csv
from datetime import datetime
import time
import tkinter as tk
from tkinter import messagebox

# ----------------- Tkinter GUI for Gesture Name -----------------
gesture_label = None

def start_program():
    global gesture_label
    gesture_label = entry.get().strip()
    
    if not gesture_label:
        messagebox.showerror("Error", "Please enter a gesture name!")
        return
    
    root.destroy()

root = tk.Tk()
root.title("Gesture Label Input")
root.geometry("400x200")
root.configure(bg="#2c2c2c")

label_font = ("Arial", 16)
entry_font = ("Arial", 14)
button_font = ("Arial", 12)

label = tk.Label(root, text="Enter Gesture Name:", fg="white", bg="#2c2c2c", font=label_font)
label.pack(pady=10)

entry = tk.Entry(root, font=entry_font, width=25)
entry.pack(pady=10)

start_button = tk.Button(root, text="Start Recording", command=start_program, font=button_font, bg="#4CAF50", fg="white", activebackground="#45a049")
start_button.pack(pady=15)

root.mainloop()

# ----------------- Main Recording Setup -----------------
SAVE_RAW_VIDEO = True
OUTPUT_FOLDER = "data/processed_csv"
RAW_VIDEO_FOLDER = "data/raw_videos"
DURATION_SEC = 30

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if SAVE_RAW_VIDEO:
    os.makedirs(RAW_VIDEO_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()
#This code is written by Aditya Sinha https://www.linkedin.com/in/adityasinha2006/
print("[INFO] Press 's' to start recording or 'q' to quit.")

# ----------------- Wait for Start/Exit Key -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv.flip(frame, 1)
    cv.putText(frame, "Press 's' to start recording or 'q' to quit.", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv.imshow("Gesture Preview", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        print("[INFO] Recording will start in 2 seconds...")
        time.sleep(2)
        print("[INFO] Recording started...")
        break
    if key == ord('q'):
        print("[INFO] Exiting...")
        cap.release()
        hands.close()
        cv.destroyAllWindows()
        exit()

# ----------------- File Naming -----------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if SAVE_RAW_VIDEO:
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_filename = os.path.join(RAW_VIDEO_FOLDER, f"{gesture_label}_{timestamp}.mp4")
    out = cv.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

columns = (
    [f"x{i}_h1" for i in range(1, 22)] + [f"y{i}_h1" for i in range(1, 22)] + [f"z{i}_h1" for i in range(1, 22)] +
    [f"x{i}_h2" for i in range(1, 22)] + [f"y{i}_h2" for i in range(1, 22)] + [f"z{i}_h2" for i in range(1, 22)]
)

csv_filename = os.path.join(OUTPUT_FOLDER, f"{gesture_label}_{timestamp}.csv")

# ----------------- Recording Loop -----------------
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)

    start_time = cv.getTickCount()
    fps = cv.getTickFrequency()

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
        writer.writerow(combined_row)

        if SAVE_RAW_VIDEO:
            out.write(frame)

        cv.imshow("Gesture Recording", frame)

        elapsed_time = (cv.getTickCount() - start_time) / fps
        if elapsed_time > DURATION_SEC:
            print("[INFO] Recording complete.")
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Recording cancelled.")
            break

# ----------------- Cleanup -----------------
cap.release()
hands.close()
if SAVE_RAW_VIDEO:
    out.release()
cv.destroyAllWindows()

print(f"[INFO] Landmark data saved to: {csv_filename}")

# ----------------- Tkinter Popup - Collection Done -----------------
popup = tk.Tk()
popup.title("Recording Complete")
popup.geometry("400x150")
popup.configure(bg="#2c2c2c")

msg = tk.Label(popup, text="✅ Gesture collection completed successfully!", fg="white", bg="#2c2c2c", font=("Arial", 14))
msg.pack(pady=20)

ok_button = tk.Button(popup, text="OK", command=popup.destroy, font=("Arial", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
ok_button.pack(pady=10)

popup.mainloop()
