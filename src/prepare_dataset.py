from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_PATHS = [
    PROJECT_ROOT / "dataset" / "train",
    PROJECT_ROOT / "dataset" / "test"
]

OUTPUT_PATH = PROJECT_ROOT / "dataset" / "images"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    total = 0

    for source_path in SOURCE_PATHS:
        if not source_path.exists():
            print(f"Skipping missing folder: {source_path}")
            continue

        for class_folder in source_path.iterdir():
            if not class_folder.is_dir():
                continue

            label = class_folder.name.upper()
            destination_folder = OUTPUT_PATH / label
            destination_folder.mkdir(parents=True, exist_ok=True)

            for image_path in class_folder.iterdir():
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                destination = destination_folder / image_path.name

                # Avoid overwriting if the same filename exists
                # in train and test.
                if destination.exists():
                    stem = image_path.stem
                    suffix = image_path.suffix
                    counter = 1

                    while destination.exists():
                        destination = (
                            destination_folder
                            / f"{stem}_{counter}{suffix}"
                        )
                        counter += 1

                shutil.copy2(image_path, destination)

                total += 1

    print("\n==============================")
    print("DATASET PREPARATION COMPLETE")
    print("==============================")
    print(f"Images copied: {total}")
    print(f"Combined dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

