from pathlib import Path
import zipfile
import os
from datasets import load_dataset
from huggingface_hub import HfApi
# import kagglehub

from app.config import settings


# ============================================================
# HUGGING FACE
# ============================================================

def search_huggingface(query, limit=5):
    """Search Hugging Face datasets."""

    api = HfApi(token=settings.hf_token)

    print("\nSearching Hugging Face...")

    datasets = list(
        api.list_datasets(
            search=query,
            limit=limit,
        )
    )

    results = []

    for dataset in datasets:
        results.append({
            "source": "Hugging Face",
            "id": dataset.id,
            "downloads": dataset.downloads or 0,
            "likes": dataset.likes or 0,
        })

    return results


# ============================================================
# KAGGLE
# ============================================================
# def search_kaggle(query, limit=5):
#     """
#     Search Kaggle datasets using credentials
#     loaded from .env.
#     """

#     print("Searching Kaggle...")

#     try:
#         # Pass .env credentials to Kaggle
#         os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        

#         from kaggle.api.kaggle_api_extended import KaggleApi

#         api = KaggleApi()
#         api.authenticate()

#         datasets = api.dataset_list(
#             search=query,
#             max_size=limit,
#         )

#         results = []

#         for dataset in datasets:

#             results.append({
#                 "source": "Kaggle",
#                 "id": dataset.ref,
#                 "downloads": getattr(
#                     dataset,
#                     "download_count",
#                     0,
#                 ),
#                 "likes": getattr(
#                     dataset,
#                     "vote_count",
#                     0,
#                 ),
#             })

#         return results

#     except Exception as e:
#         print(f"Kaggle search error: {e}")
#         return []

# ============================================================
# COMBINED SEARCH
# ============================================================

def search_all_sources(query):
    """Search Hugging Face and Kaggle."""

    hf_results = search_huggingface(query)
    # kaggle_results = search_kaggle(query)

    return hf_results 

# ============================================================
# DISPLAY RESULTS
# ============================================================

def show_candidates(candidates):

    if not candidates:

        print("\nNo datasets found.")

        return

    print("\n========================================")
    print("         DATASET CANDIDATES")
    print("========================================\n")

    for index, dataset in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"{index}. [{dataset['source']}] "
            f"{dataset['id']}"
        )

        print(
            f"   Downloads: {dataset['downloads']}"
        )

        print(
            f"   Likes/Votes: {dataset['likes']}"
        )

        print()


# ============================================================
# CHOOSE DATASET
# ============================================================

def choose_dataset(candidates):

    while True:

        try:

            choice = int(
                input(
                    f"Select dataset "
                    f"(1-{len(candidates)}): "
                )
            )

            if 1 <= choice <= len(candidates):

                return candidates[choice - 1]

            print(
                f"Please select between 1 and "
                f"{len(candidates)}."
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# SAMPLE SIZE
# ============================================================

def ask_sample_size():

    while True:

        try:

            sample_size = int(
                input(
                    "\nHow many samples do you need? "
                )
            )

            if sample_size <= 0:

                print(
                    "Sample size must be greater than 0."
                )

                continue

            return sample_size

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# HUGGING FACE DATASET
# ============================================================

def load_huggingface_dataset(dataset_id):

    print(
        f"\nLoading Hugging Face dataset:"
        f"\n{dataset_id}"
    )

    dataset = load_dataset(
        dataset_id,
        split="train",
        token=settings.hf_token,
    )

    return dataset


# ============================================================
# KAGGLE DATASET
# ============================================================

# def download_kaggle_dataset(dataset_id):
#     import os
#     from pathlib import Path
#     from kaggle.api.kaggle_api_extended import KaggleApi

#     print(f"\nDownloading Kaggle dataset:")
#     print(dataset_id)

#     try:
#         os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        

#         api = KaggleApi()
#         api.authenticate()

#         output_dir = Path("data/raw/kaggle")
#         output_dir.mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         api.dataset_download_files(
#             dataset_id,
#             path=str(output_dir),
#             unzip=True,
#         )

#         print("✓ Kaggle dataset downloaded.")

#         return output_dir

#     except Exception as e:
#         print(f"\nKaggle download failed: {e}")
#         return None
        
        
# ============================================================
# LABEL DETECTION
# ============================================================

def detect_label_feature(dataset):

    for name, feature in dataset.features.items():

        if hasattr(feature, "names"):

            if feature.names:

                return name, feature

    return None, None


# ============================================================
# TARGET CLASS
# ============================================================

def ask_target_class(label_feature):

    print("\nAvailable classes:\n")

    for index, label in enumerate(
        label_feature.names
    ):

        print(
            f"{index}: {label}"
        )

    target = input(
        "\nWhat class/label do you need? "
    ).strip()

    return target


# ============================================================
# FILTER
# ============================================================

def filter_by_class(
    dataset,
    label_name,
    label_feature,
    target_class,
):

    try:

        target_id = label_feature.str2int(
            target_class
        )

    except ValueError:

        print(
            f"\nClass '{target_class}' "
            "was not found."
        )

        return None

    print(
        f"\nFiltering for class: "
        f"{target_class}"
    )

    filtered = dataset.filter(
        lambda example:
            example[label_name] == target_id
    )

    return filtered


# ============================================================
# SAMPLE SELECTION
# ============================================================

def select_samples(
    dataset,
    sample_size,
):

    available = len(dataset)

    print(
        f"\nAvailable matching samples: "
        f"{available}"
    )

    if sample_size > available:

        print(
            f"Only {available} matching "
            f"samples are available."
        )

        return None

    return dataset.select(
        range(sample_size)
    )


# ============================================================
# ZIP
# ============================================================

def create_zip(
    dataset,
    target_name,
    sample_size,
):

    output_dir = Path(
        f"data/processed/"
        f"{target_name.lower()}_{sample_size}"
    )

    zip_path = Path(
        f"data/processed/"
        f"{target_name.lower()}_{sample_size}.zip"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nExporting images...")

    for index, example in enumerate(dataset):

        image = example["image"]

        if image.mode != "RGB":

            image = image.convert("RGB")

        image_path = (
            output_dir
            / f"{target_name}_{index:04d}.jpg"
        )

        image.save(
            image_path,
            format="JPEG",
        )

    print(
        f"Exported {len(dataset)} images."
    )

    print("Creating ZIP...")

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for image_path in output_dir.iterdir():

            if image_path.is_file():

                zip_file.write(
                    image_path,
                    arcname=image_path.name,
                )

    return zip_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("              DataR²")
    print("       Data Request & Retrieval")
    print("========================================")

    query = input(
        "\nWhat dataset do you need?\n> "
    ).strip()

    if not query:

        print(
            "Dataset request cannot be empty."
        )

        return

    # --------------------------------------------------------
    # SEARCH BOTH SOURCES
    # --------------------------------------------------------

    candidates = search_all_sources(query)

    if not candidates:

        print(
            "\nNo datasets found on "
            "Hugging Face or Kaggle."
        )

        return

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    show_candidates(candidates)

    # --------------------------------------------------------
    # USER SELECTS
    # --------------------------------------------------------

    selected = choose_dataset(
        candidates
    )

    source = selected["source"]
    dataset_id = selected["id"]

    print(
        f"\nSelected:"
        f"\nSource: {source}"
        f"\nDataset: {dataset_id}"
    )

    # --------------------------------------------------------
    # SAMPLE SIZE
    # --------------------------------------------------------

    sample_size = ask_sample_size()

    # --------------------------------------------------------
    # HUGGING FACE
    # --------------------------------------------------------

    if source == "Hugging Face":

        dataset = load_huggingface_dataset(
            dataset_id
        )

        print(
            f"\nTotal dataset samples: "
            f"{len(dataset)}"
        )

        label_name, label_feature = (
            detect_label_feature(dataset)
        )

        if label_feature is None:

            print(
                "\nCould not automatically "
                "detect a class label."
            )

            return

        print(
            f"\nDetected label field: "
            f"{label_name}"
        )

        target_class = ask_target_class(
            label_feature
        )

        filtered_dataset = filter_by_class(
            dataset,
            label_name,
            label_feature,
            target_class,
        )

        if filtered_dataset is None:

            return

        selected_dataset = select_samples(
            filtered_dataset,
            sample_size,
        )

        if selected_dataset is None:

            return

        zip_path = create_zip(
            selected_dataset,
            target_class,
            sample_size,
        )

        print("\n========================================")
        print("             ✓ COMPLETE")
        print("========================================")
        print(f"Source:       {source}")
        print(f"Dataset:      {dataset_id}")
        print(f"Class:        {target_class}")
        print(f"Samples:      {sample_size}")
        print(f"ZIP:          {zip_path}")
        print("========================================")

    # --------------------------------------------------------
    # KAGGLE
    # --------------------------------------------------------

    # elif source == "Kaggle":

    #     path = download_kaggle_dataset(
    #         dataset_id
    #     )

    #     if path is None:

    #         return

    #     print("\n========================================")
    #     print("       KAGGLE DATASET ACQUIRED")
    #     print("========================================")
    #     print(f"Source:       Kaggle")
    #     print(f"Dataset:      {dataset_id}")
    #     print(f"Requested:    {sample_size}")
    #     print(f"Downloaded:   {path}")
    #     print("========================================")

    #     print(
    #         "\nKaggle processing will be "
    #         "implemented next."
    #     )


if __name__ == "__main__":
    main()
