from huggingface_hub import HfApi

from backend.app.config import settings


api = HfApi(token=settings.hf_token)

dataset_ids = [
    "Marxulia/asl_sign_languages_alphabets_v02",
    "Marxulia/asl_sign_languages_alphabets_v03",
    "Katyaaaaae/asl-alphabet",
    "thels07/asl_sign_languages_alphabets_v03",
]


for dataset_id in dataset_ids:
    print("\n" + "=" * 70)
    print(dataset_id)
    print("=" * 70)

    try:
        info = api.dataset_info(dataset_id)

        print("Author:", info.author)
        print("Downloads:", info.downloads)
        print("Likes:", info.likes)
        print("Tags:", info.tags)

        if info.card_data:
            print("Card data:")
            print(info.card_data)

    except Exception as e:
        print("ERROR:", e)
