from datasets import load_dataset

from backend.app.config import settings


DATASET_ID = "Marxulia/asl_sign_languages_alphabets_v03"
TARGET_CLASS = "Z"


print("Loading dataset...")

dataset = load_dataset(
    DATASET_ID,
    split="train",
    token=settings.hf_token,
)

label_feature = dataset.features["label"]

target_id = label_feature.str2int(TARGET_CLASS)

print(f"Target class: {TARGET_CLASS}")
print(f"Target label ID: {target_id}")

z_dataset = dataset.filter(
    lambda example: example["label"] == target_id
)

print("\nFiltering complete!")
print("Total samples:", len(dataset))
print("Z samples:", len(z_dataset))
