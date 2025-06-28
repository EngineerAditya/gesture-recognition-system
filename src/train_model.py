# train_model.py
# Author: Aditya Sinha
# Purpose: Train a baseline gesture recognition model using scikit-learn
# Notes: Label encoding & saving model included

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------- Load Combined Dataset ---------
print("[INFO] Loading gesture dataset...")
DATASET_PATH = "data/combined_csv/combined_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# --------- Feature & Label Separation ---------
X = df.drop("label", axis=1)
y = df["label"]

# --------- Encode Labels ---------
print("[INFO] Encoding gesture labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Optional: view mapping
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"[INFO] Label Map: {label_map}")

# --------- Train-Test Split ---------
print("[INFO] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --------- Train Model ---------
print("[INFO] Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --------- Evaluate Model ---------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------- Save Model & Label Encoder ---------
print("[INFO] Saving model and label encoder...")
os.makedirs("models", exist_ok=True)
with open("models/sk_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model saved to models/sk_model.pkl")
print("✅ Label encoder saved to models/label_encoder.pkl")
