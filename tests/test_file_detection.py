import pytest
import asyncio
from pathlib import Path
import tempfile
from utils.file_detector import detect_file_type


@pytest.mark.asyncio
async def test_pdf_detection(tmp_path: Path):
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    out = await detect_file_type(p)
    assert out["type"] == "pdf"


@pytest.mark.asyncio
async def test_sqlite_detection(tmp_path: Path):
    p = tmp_path / "db.sqlite"
    p.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    out = await detect_file_type(p)
    assert out["type"] == "sqlite"


@pytest.mark.asyncio
async def test_csv_detection(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("col1,col2\n1,2\n3,4\n")
    out = await detect_file_type(p)
    assert out["type"] == "csv"


@pytest.mark.asyncio
async def test_zip_detection(tmp_path: Path):
    p = tmp_path / "archive.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    out = await detect_file_type(p)
    # zip fallback
    assert out["type"] in ("zip", "excel")


@pytest.mark.asyncio
async def test_json_detection(tm
