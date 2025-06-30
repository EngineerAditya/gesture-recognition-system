# gesture-recognition-system

# 🤖 Real-Time Hand Gesture Recognition with Voice Feedback

This project uses **MediaPipe**, **TensorFlow**, and **OpenCV** to recognize hand gestures from a webcam in real-time. It also provides **audio feedback** using a Text-to-Speech (TTS) engine to announce the detected gesture.

Watch the code run here: https://www.linkedin.com/posts/adityasinha2006_machinelearning-computervision-ai-activity-7345395881346527233-9J2r?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFIBTYABLifzvbm0bR14dN22zxSvlZjLNCs

---

## 🎯 Project Features

👉 Real-time hand gesture detection using your webcam  
👉 Detects both left and right hands  
👉 Classifies gestures using a trained deep learning model (TensorFlow)  
👉 Announces detected gestures out loud with offline TTS  
👉 Compatible with **Windows**, **Linux**, and **macOS** (Linux requires `espeak`)  
👉 Cooldown logic prevents repeated announcements when the same gesture is held  
👉 Well-structured, beginner-friendly, and easy-to-understand code  
👉 `data_app.py` automates the complete pipeline — collects data, prepares dataset, trains model — in order  

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/EngineerAditya/gesture-recognition-system.git
cd gesture-recognition-system
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Additional Setup for Linux Users (Text-to-Speech)

On Linux, `pyttsx3` relies on `espeak` for speech synthesis. Install it using:

```bash
sudo apt update
sudo apt install espeak
```

### 4. Collect Data, Prepare Dataset & Train the Model (One-Click)

Simply run:

```bash
python data_app.py
```

This will:

✅ Launch the data collection script  
✅ Automatically combine the dataset  
✅ Train the TensorFlow model and save it  

Your trained model (`models/tf_gesture_model.h5`) and label encoder (`models/label_encoder.pkl`) will be ready for use.

---

## 🎮 Running Real-Time Gesture Recognition

Make sure your webcam is working and run:

```bash
python predict_live.py
```

### Controls:

- Press **'q'** to quit  
- Detected gestures will appear on the screen  
- The gesture will be spoken aloud using TTS  

---

## 🔊 Text-to-Speech Details

- Uses `pyttsx3` for offline TTS (cross-platform)  
- On Linux, requires `espeak` system package  
- Cooldown of **3 seconds** prevents constant repetition  
- Announces new gestures immediately  

---

## 📦 Project Structure

```
gesture-recognition-system/
├── data/                  # Collected data and processed CSVs
├── models/                # Saved TensorFlow model & label encoder
├── src/                   # Data collection & preparation scripts
├── data_app.py            # Automates collect, prepare & train steps
├── predict_live.py        # Real-time gesture recognition script
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
```

---

## 🧑‍💻 Technologies Used

- [MediaPipe](https://developers.google.com/mediapipe) — Hand landmark detection  
- [OpenCV](https://opencv.org/) — Video processing  
- [TensorFlow](https://www.tensorflow.org/) — Gesture classification  
- [pyttsx3](https://pypi.org/project/pyttsx3/) — Offline text-to-speech  
- [Pandas & NumPy](https://pandas.pydata.org/) — Data handling  

---

## 💡 Future Improvements

- More complex deep learning models for higher accuracy  
- Larger, more diverse gesture dataset  
- Optional integration with `gTTS` for natural-sounding speech (requires internet)  
- GUI-based interaction for non-technical users  

---

## ✨ Credits

I have used **ChatGPT** to structure and make this project code clean, readable, and beginner-friendly. 😍

---

## 🤝 Contributions

Feel free to fork, suggest improvements, or open pull requests. All contributions are welcome!
