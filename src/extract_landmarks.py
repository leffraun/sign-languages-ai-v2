import mediapipe as mp
import cv2
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "images"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUTPUT_PATH = PROJECT_ROOT / "landmarks" / "landmarks.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

header = ["label"]

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

total_images = 0
successful_images = 0
failed_images = 0

with HandLandmarker.create_from_options(options) as landmarker:

    with open(OUTPUT_PATH, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(header)

        for class_folder in sorted(DATASET_PATH.iterdir()):

            if not class_folder.is_dir():
                continue

            label = class_folder.name

            print(f"\nProcessing class: {label}")

            images = [
                file
                for file in class_folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            for image_path in images:

                total_images += 1

                try:

                    image = cv2.imread(str(image_path))

                    if image is None:
                        failed_images += 1
                        continue

                    image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image
                    )

                    result = landmarker.detect(mp_image)

                    if not result.hand_landmarks:
                        failed_images += 1
                        continue

                    hand = result.hand_landmarks[0]

                    row = [label]

                    for landmark in hand:
                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    writer.writerow(row)

                    successful_images += 1

                except Exception as e:

                    failed_images += 1

                    print(
                        f"Error processing {image_path}: {e}"
                    )

                if total_images % 100 == 0:

                    print(
                        f"Images processed: {total_images} | "
                        f"Successful: {successful_images} | "
                        f"Failed: {failed_images}"
                    )


print("\n==============================")
print("EXTRACTION COMPLETE")
print("==============================")
print(f"Total images:      {total_images}")
print(f"Successful:        {successful_images}")
print(f"Failed/skipped:    {failed_images}")
print(f"CSV saved to:      {OUTPUT_PATH}")
