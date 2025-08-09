from pathlib import Path
import os

# Small helper file — storage is local filesystem for now. Keep abstraction for S3 later.

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_upload_path(dataset_id: str, original_name: str) -> Path:
    # ensure we keep extension but prevent path traversals
    name = Path(original_name).name
    suffix = Path(name).suffix
    return UPLOAD_DIR / f"{dataset_id}{suffix}"
