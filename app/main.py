import os
import uuid
import time
import logging
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import aiofiles
from utils.file_detector import detect_file_type
from metadata import write_metadata

# Logging setup (structured-ish)
logger = logging.getLogger("upload_service")
handler = logging.StreamHandler()
formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="File Upload Pipeline")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
METADATA_DIR = Path(os.getenv("METADATA_DIR", "metadata"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_BYTES", 1024 * 1024 * 1024))  # 1GB default
CHUNK_SIZE = 1024 * 1024

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    detected_type: str
    size: int
    storage_path: str
    uploaded_at: float


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # sanitize filename
    original_filename = Path(file.filename).name
    if not original_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    dataset_id = str(uuid.uuid4())
    storage_name = f"{dataset_id}{Path(original_filename).suffix}"
    storage_path = UPLOAD_DIR / storage_name

    # Stream write to disk
    size = 0
    try:
        async with aiofiles.open(storage_path, "wb") as out_f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    # remove partial file
                    await out_f.close()
                    try:
                        storage_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail=f"File exceeds configured max size ({MAX_FILE_SIZE} bytes)")
                await out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to write upload")
        raise HTTPException(status_code=500, detail="Failed to save file")
    finally:
        # ensure UploadFile's internal file-like is closed
        try:
            await file.close()
        except Exception:
            pass

    # Detect file type using heuristics + libmagic
    try:
        async_detect = await detect_file_type(storage_path)
        detected_type = async_detect.get("type", "unknown")
        detect_meta = async_detect
    except Exception as e:
        logger.exception("Detection failed")
        detected_type = "unknown"
        detect_meta = {}

    # prepare metadata
    metadata = {
        "dataset_id": dataset_id,
        "original_filename": original_filename,
        "detected_type": detected_type,
        "detection": detect_meta,
        "size": size,
        "storage_path": str(storage_path.resolve()),
        "uploaded_at": time.time(),
    }

    # write metadata file
    try:
        write_metadata(METADATA_DIR / f"{dataset_id}.json", metadata)
    except Exception:
        logger.exception("Failed to write metadata")
        # non-fatal to upload success

    logger.info(f"Uploaded {original_filename} -> {storage_path} (type={detected_type}, size={size})")

    return UploadResponse(
        dataset_id=dataset_id,
        filename=original_filename,
        detected_type=detected_type,
        size=size,
        storage_path=str(storage_path),
        uploaded_at=metadata["uploaded_at"],
    )
