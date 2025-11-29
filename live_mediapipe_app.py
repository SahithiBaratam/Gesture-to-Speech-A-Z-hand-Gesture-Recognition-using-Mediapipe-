import cv2
import mediapipe as mp
import numpy as np
import joblib

from sentence_builder import add_letter, add_space, backspace, clear_sentence, get_sentence
from tts_engine import speak

MODEL_PATH = "../models/mp_model.pkl"
ENCODER_PATH = "../models/mp_label_encoder.pkl"

mp_hands = mp.solutions.hands

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

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

    last_letter = ""
    stable_count = 0
    STAB = 5
    current_letter = "-"

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

            detected = "-"

            if landmarks is not None:
                pred = model.predict([landmarks])[0]
                letter = label_encoder.inverse_transform([pred])[0]

                if letter == last_letter:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_letter = letter

                if stable_count > STAB:
                    detected = letter
                    current_letter = letter

            cv2.putText(frame, f"Letter: {detected}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Sentence: {get_sentence()}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(frame,
                        "A:Add  Space:Space  B:Backspace  C:Clear  S:Speak  ESC:Exit",
                        (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,0), 1)

            cv2.imshow("Mediapipe Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if key in (ord('a'), ord('A')):
                add_letter(current_letter)
            if key == 32:
                add_space()
            if key in (ord('b'), ord('B')):
                backspace()
            if key in (ord('c'), ord('C')):
                clear_sentence()
            if key in (ord('s'), ord('S')):
                speak(get_sentence())

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 