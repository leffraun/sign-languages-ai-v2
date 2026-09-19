import cv2
import mediapipe as mp
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "bsl_sign_language_rf.pkl"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


# -----------------------------
# Load trained Random Forest
# -----------------------------

model = joblib.load(MODEL_PATH)


# -----------------------------
# Set up MediaPipe
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)


# -----------------------------
# Start webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()


with HandLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break


        # OpenCV uses BGR.
        # MediaPipe expects RGB.

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Convert frame to MediaPipe image

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # Detect hand

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # If a hand was detected

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Extract x, y, z values

            features = []

            for landmark in hand:

                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])


            # Convert to NumPy array

            features = np.array(features).reshape(1, -1)


            # Predict letter

            feature_names = []

            for i in range(21):
                feature_names.extend([f"x{i}", f"y{i}", f"z{i}"])

            features_df = pd.DataFrame(features, columns=feature_names)

            prediction = model.predict(features_df)


            # Display prediction

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )


        else:

            cv2.putText(
                frame,
                "No hand detected",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # Show webcam

        cv2.imshow(
            "BSL Fingerspelling Recognition",
            frame
        )


        # Press Q to quit

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
