# Gesture-to-Speech(A-Z-hand-Gesture-Recognition-using-Mediapipe)
A real-time gesture recognition and speech generation system designed for non-verbal patients. This project detects A–Z hand gestures using MediaPipe, converts them into text, and finally produces speech output.
The system is lightweight, accurate, and runs completely offline.

🚀 Features
✔ Real-time gesture recognition using MediaPipe Hands
✔ Classifies A–Z alphabets
✔ Builds sentences from detected characters
✔ Converts text to speech using pyttsx3
✔ Offline — no internet required
✔ Fast and lightweight (no TensorFlow needed)
✔ Works on any background and lighting conditions
✔ Easy to train with your own gesture samples

📌 Project Structure
Gesture_Mediapipe/
│
├── data/
│     └── mp_landmarks.csv             # Auto-generated gesture dataset
│
├── models/
│     ├── mp_model.pkl                 # Trained RandomForest model
│     └── mp_label_encoder.pkl         # Label encoder for A–Z
│
├── src/
│     ├── collect_mediapipe_data.py    # Collect hand gesture samples
│     ├── train_mediapipe_model.py     # Train the ML model
│     ├── live_mediapipe_app.py        # Main real-time recognition app
│     ├── sentence_builder.py          # Handles sentence construction
│     ├── tts_engine.py                # Text-to-Speech engine (multiple speak fix)
│
├── venv/                               # Virtual environment (not uploaded)
│
└── README.md

🛠 Installation
1️⃣ Clone the repository
git clone <your-repo-link>
cd Gesture_Mediapipe
2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
3️⃣ Install dependencies
pip install -r requirements.txt


🧪 Step 1 — Collect Gesture Data
cd src
python collect_mediapipe_data.py
Controls:
| Action                   | Key                     |
| ------------------------ | ----------------------- |
| Save sample for a letter | Press that letter (A–Z) |
| Quit                     | **ESC**                 |
👉 Collect at least 20–40 samples per letter for good accuracy.
A file will be created:
data/mp_landmarks.csv

🧠 Step 2 — Train the Model
python train_mediapipe_model.py
Outputs:
models/mp_model.pkl
models/mp_label_encoder.pkl

🎤 Step 3 — Run the Real-Time Gesture-to-Speech App
python live_mediapipe_app.py
| Action              | Key       |
| ------------------- | --------- |
| Add detected letter | **A**     |
| Add space           | **SPACE** |
| Backspace           | **B**     |
| Clear sentence      | **C**     |
| Speak full sentence | **S**     |
| Exit application    | **ESC**   |

🧠 How It Works
1. MediaPipe Hands extracts 21 hand landmarks
2. Landmarks are flattened into a 63-point feature vector
3. A RandomForest model predicts the gesture (A–Z)
4. The predicted letter is added to your live sentence
5. Sentence is spoken out loud via pyttsx3


✨ Advantages
Offline
Fast prediction
Works with any webcam
Easy to extend (digits, words, custom gestures)
No deep learning required


🚧 Future Enhancements
Tkinter GUI interface
Dynamic gesture support
Predefined common phrases (“Help me”, “Water”, etc.)
Multilingual speech output
Android version using MediaPipe + TFLite


🤝 Contributing
Contributions are welcome!
You may:
Add new gestures
Improve UI
Enhance accuracy
Document the project


📜 License
This project is open source.
You may modify and reuse it for academic or personal use.


❤️ Acknowledgements
1. MediaPipe by Google
2. OpenCV
3. scikit-learn
4. pyttsx3

<img width="794" height="641" alt="Screenshot 2025-11-28 080709" src="https://github.com/user-attachments/assets/e4ca8c36-375b-434b-9be8-379b5a27f9f6" />
<img width="795" height="634" alt="Screenshot 2025-11-28 080647" src="https://github.com/user-attachments/assets/9ced98f2-ecfb-4485-895b-d6c517d8196d" />

