# I have used ChatGPT to structure and make this code readable 😄
# Purpose: Train gesture recognition model using TensorFlow, Label encoding & model saving included

import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------- Load Combined Dataset -----------------
print("[INFO] Loading gesture dataset...")
DATASET_PATH = "data/combined_csv/combined_dataset.csv"

df = pd.read_csv(DATASET_PATH)
print(f"[INFO] Dataset shape: {df.shape}")

# ----------------- Feature & Label Separation -----------------
print("[INFO] Splitting features and labels...")
X = df.drop("label", axis=1).values
y = df["label"].values

# ----------------- Encode Labels -----------------
print("[INFO] Encoding gesture labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Optional: view label-to-number mapping
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"[INFO] Label Mapping: {label_map}")

# ----------------- Train-Test Split -----------------
print("[INFO] Creating train-test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ----------------- One-hot Encode Labels -----------------
num_classes = len(np.unique(y_encoded))
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)

# ----------------- Build TensorFlow Model -----------------
print("[INFO] Building the gesture classification model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------- Train the Model -----------------
print("[INFO] Starting model training...")
history = model.fit(
    X_train, y_train_categorical,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ----------------- Evaluate Model -----------------
print("[INFO] Evaluating model on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# ----------------- Detailed Classification Report -----------------
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------- Save Model & Label Encoder -----------------
print("[INFO] Saving trained model and label encoder...")
os.makedirs("models", exist_ok=True)

model.save("models/tf_gesture_model.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅ Model saved to: models/tf_gesture_model.h5")
print("✅ Label encoder saved to: models/label_encoder.pkl")
