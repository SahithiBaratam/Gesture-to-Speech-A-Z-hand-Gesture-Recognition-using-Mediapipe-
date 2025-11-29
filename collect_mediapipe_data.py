import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "mp_landmarks.csv")

mp_hands = mp.solutions.hands

def extract_landmarks(image, hands):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    
    coords = []
    for lm in result.multi_hand_landmarks[0].landmark:
        coords.extend([lm.x, lm.y, lm.z])
    
    return np.array(coords, dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0)
    all_rows = []

    print("===============================================")
    print(" MEDIAPIPE DATA COLLECTION")
    print(" Show a gesture → Press A-Z to save sample")
    print(" Press ESC to finish and save CSV")
    print("===============================================")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks = extract_landmarks(frame, hands)

            if landmarks is not None:
                cv2.putText(frame, "Hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "No hand", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Collect Mediapipe Data", frame)

            key = cv2.waitKey(1) & 0xFF

            # ESC quits
            if key == 27:
                break

            # A–Z key saves sample
            if ord('a') <= key <= ord('z'):
                label = chr(key).upper()
                if landmarks is not None:
                    row = np.append(landmarks, label)
                    all_rows.append(row)
                    print(f"[SAVED] {label} (total samples: {len(all_rows)})")
                else:
                    print("[WARN] No hand detected → sample not saved")

    cap.release()
    cv2.destroyAllWindows()

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(CSV_PATH, index=False)
        print(f"[DONE] Saved {len(all_rows)} samples to {CSV_PATH}")
    else:
        print("[INFO] No samples collected.")

if __name__ == "__main__":
    main()
