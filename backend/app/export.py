from pathlib import Path
import zipfile

from datasets import load_dataset

from backend.app.config import settings


DATASET_ID = "Marxulia/asl_sign_languages_alphabets_v03"
TARGET_CLASS = "Z"

OUTPUT_DIR = Path("data/processed/asl_z")
ZIP_PATH = Path("data/processed/asl_z.zip")


print("Loading dataset...")

dataset = load_dataset(
    DATASET_ID,
    split="train",
    token=settings.hf_token,
)

label_feature = dataset.features["label"]
target_id = label_feature.str2int(TARGET_CLASS)

z_dataset = dataset.filter(
    lambda example: example["label"] == target_id
)

print(f"Found {len(z_dataset)} Z samples.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Exporting images...")

for index, example in enumerate(z_dataset):
    image = example["image"]

    image_path = OUTPUT_DIR / f"Z_{index:04d}.jpg"

    # Ensure consistent image format
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(image_path, format="JPEG")

print(f"Exported {len(z_dataset)} images.")

print("Creating ZIP...")

ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as zip_file:

    for image_path in OUTPUT_DIR.iterdir():
        if image_path.is_file():
            zip_file.write(
                image_path,
                arcname=image_path.name,
            )

print(f"\n✓ Done!")
print(f"Images: {OUTPUT_DIR}")
print(f"ZIP: {ZIP_PATH}")
