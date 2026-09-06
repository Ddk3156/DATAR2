from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.menu import (
    search_all_sources,
    load_huggingface_dataset,
    detect_label_feature,
    filter_by_class,
    select_samples,
    create_zip,
)

app = FastAPI(
    title="DataR² API",
    description="Data Requirement & Retrieval API",
    version="1.0.0",
)

FRONTEND_FILE = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

# Latest search results
search_results = []


class SearchRequest(BaseModel):
    dataset: str


class ProcessRequest(BaseModel):
    dataset_index: int
    sample_size: int
    target_class: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_FILE)


# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------

@app.post("/retrieve")
def retrieve_dataset(request: SearchRequest):

    global search_results

    search_results = search_all_sources(request.dataset)

    # Hugging Face only
    search_results = [
        result
        for result in search_results
        if result["source"] == "Hugging Face"
    ]

    return {
        "status": "success",
        "results": search_results,
    }


# ---------------------------------------------------------
# CHECK SELECTED DATASET
# ---------------------------------------------------------

@app.post("/inspect")
def inspect_dataset(request: ProcessRequest):

    if not search_results:
        return {
            "status": "error",
            "message": "Please search for a dataset first."
        }

    if request.dataset_index < 1 or request.dataset_index > len(search_results):
        return {
            "status": "error",
            "message": "Invalid dataset selection."
        }

    selected = search_results[request.dataset_index - 1]
    dataset_id = selected["id"]

    try:

        dataset = load_huggingface_dataset(dataset_id)

        label_name, label_feature = detect_label_feature(dataset)

        if label_name is None:
            return {
                "status": "error",
                "message": (
                    "This dataset was loaded successfully, "
                    "but no label/class feature was detected."
                ),
            }

        classes = []

        if hasattr(label_feature, "names"):
            classes = list(label_feature.names)

        return {
            "status": "success",
            "dataset": dataset_id,
            "label_name": label_name,
            "classes": classes,
        }

    except Exception as error:

        print(f"Dataset loading error: {error}")

        return {
            "status": "error",
            "message": (
                "This dataset could not be loaded using the "
                "current DataR² processing pipeline. "
                "Please choose another dataset."
            ),
        }


# ---------------------------------------------------------
# PROCESS DATASET
# ---------------------------------------------------------

@app.post("/process")
def process_dataset(request: ProcessRequest):

    if not search_results:
        return {
            "status": "error",
            "message": "Please search for a dataset first."
        }

    if request.dataset_index < 1 or request.dataset_index > len(search_results):
        return {
            "status": "error",
            "message": "Invalid dataset selection."
        }

    if request.sample_size <= 0:
        return {
            "status": "error",
            "message": "Sample size must be greater than zero."
        }

    selected = search_results[request.dataset_index - 1]
    dataset_id = selected["id"]

    try:

        # Load dataset
        dataset = load_huggingface_dataset(dataset_id)

        # Detect label
        label_name, label_feature = detect_label_feature(dataset)

        if label_name is None:
            return {
                "status": "error",
                "message": "No label/class feature was detected."
            }

        # Filter class
        filtered_dataset = filter_by_class(
            dataset,
            label_name,
            label_feature,
            request.target_class,
        )

        available_samples = len(filtered_dataset)

        if available_samples == 0:
            return {
                "status": "error",
                "message": (
                    f"No samples found for class "
                    f"'{request.target_class}'."
                ),
            }

        if request.sample_size > available_samples:
            return {
                "status": "error",
                "message": (
                    f"Only {available_samples} samples are available "
                    f"for class '{request.target_class}'."
                ),
            }

        # Select samples
        samples = select_samples(
            filtered_dataset,
            request.sample_size,
        )

        # Create ZIP
        zip_path = create_zip(
            samples,
            request.target_class,
            request.sample_size,
        )

        return {
            "status": "success",
            "dataset": dataset_id,
            "class": request.target_class,
            "sample_size": request.sample_size,
            "available_samples": available_samples,
            "download_url": f"/download/{Path(zip_path).name}",
        }

    except Exception as error:

        print(f"Processing error: {error}")

        return {
            "status": "error",
            "message": (
                "Sorry, this dataset could not be processed "
                "with the current DataR² pipeline."
            ),
        }


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = Path("data/processed") / filename

    if not file_path.exists():
        return {
            "status": "error",
            "message": "File not found."
        }

    return FileResponse(
        file_path,
        media_type="application/zip",
        filename=file_path.name,
    )
