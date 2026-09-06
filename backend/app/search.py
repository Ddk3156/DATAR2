from huggingface_hub import HfApi

from backend.app.config import settings


api = HfApi(token=settings.hf_token)

datasets = api.list_datasets(
    search="ASL alphabet",
    limit=10,
)

print("\nASL dataset candidates:\n")

for dataset in datasets:
    print(dataset.id)
