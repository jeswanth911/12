import json
from typing import Any
from pathlib import Path


def write_metadata(path: Path, metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def read_metadata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
