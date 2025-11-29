import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

CSV_PATH = "../data/mp_landmarks.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "mp_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "mp_label_encoder.pkl")

def main():
    if not os.path.exists(CSV_PATH):
        print("[ERROR] No training CSV found.")
        return

    df = pd.read_csv(CSV_PATH)
    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].astype(str).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y_enc)

    preds = model.predict(X)
    acc = accuracy_score(y_enc, preds)

    print(f"[TRAIN ACCURACY] {acc*100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print("[SAVED] Model saved successfully.")

if __name__ == "__main__":
    main()
