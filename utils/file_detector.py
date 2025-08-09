import magic
from pathlib import Path
import json
import asyncio
import re
from typing import Dict

# read file header bytes async
async def _read_initial_bytes(path: Path, n: int = 8192) -> bytes:
    loop = asyncio.get_event_loop()
    def _read():
        with open(path, "rb") as f:
            return f.read(n)
    return await loop.run_in_executor(None, _read)


async def detect_file_type(path: Path) -> Dict:
    """
    Returns dict with keys: type (one of: pdf, sqlite, csv, tsv, excel, json, xml, parquet, zip, tar, sql, unknown), raw_magic, confidence
    Uses python-magic and file-signature heuristics where needed.
    """
    data = {
        "type": "unknown",
        "raw_magic": None,
        "confidence": 0.0,
    }

    initial = await _read_initial_bytes(path, 262144)

    try:
        m = magic.Magic(mime=False)
        raw = m.from_buffer(initial)
        data["raw_magic"] = raw
    except Exception:
        data["raw_magic"] = None

    # signatures
    if initial.startswith(b"%PDF-"):
        data.update({"type": "pdf", "confidence": 1.0})
        return data

    if initial.startswith(b"SQLite format 3\x00"):
        data.update({"type": "sqlite", "confidence": 1.0})
        return data

    # Parquet has magic 'PAR1' at start and end
    if initial[:4] == b"PAR1":
        data.update({"type": "parquet", "confidence": 1.0})
        return data

    # ZIP (xlsx, xlsb, zip archives)
    if initial[:4] == b"PK\x03\x04":
        # could be XLSX or ZIP
        # try to detect xlsx by file extension presence — but better: use magic
        try:
            if data["raw_magic"] and "Microsoft OOXML" in data["raw_magic"]:
                data.update({"type": "excel", "confidence": 0.95})
                return data
        except Exception:
            pass
        data.update({"type": "zip", "confidence": 0.8})
        return data

    # Text-based heuristics
    text = None
    try:
        text = initial.decode("utf-8", errors="ignore")
    except Exception:
        text = None

    if text:
        stripped = text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            # probable JSON
            try:
                json.loads(stripped[:10000])
                data.update({"type": "json", "confidence": 0.95})
                return data
            except Exception:
                pass

        # SQL dump heuristics
        if re.search(r"CREATE\s+TABLE|INSERT\s+INTO|PRAGMA|BEGIN TRANSACTION", stripped[:10000], re.IGNORECASE):
            data.update({"type": "sql", "confidence": 0.9})
            return data

        # XML
        if stripped.startswith("<") and "?>" in stripped[:100]:
            data.update({"type": "xml", "confidence": 0.9})
            return data

        # CSV/TSV heuristics — check presence of commas or tabs and consistent column counts across first 5 lines
        lines = [l for l in stripped.splitlines() if l.strip()][:20]
        if len(lines) >= 2:
            comma_counts = [line.count(",") for line in lines[:10]]
            tab_counts = [line.count("\t") for line in lines[:10]]
            if max(comma_counts) > 0 and len(set(comma_counts)) <= 3:
                data.update({"type": "csv", "confidence": 0.8})
                return data
            if max(tab_counts) > 0 and len(set(tab_counts)) <= 3:
                data.update({"type": "tsv", "confidence": 0.8})
                return data

    # Fallback on raw magic string
    if data.get("raw_magic"):
        rm = data["raw_magic"].lower()
        if "sqlite" in rm:
            data.update({"type": "sqlite", "confidence": 0.95})
            return data
        if "pdf" in rm:
            data.update({"type": "pdf", "confidence": 0.95})
            return data
        if "parquet" in rm or "apache parquet" in rm:
            data.update({"type": "parquet", "confidence": 0.95})
            return data
        if "microsoft excel" in rm or "excel" in rm or "openxml" in rm:
            data.update({"type": "excel", "confidence": 0.9})
            return data
        if "xml" in rm:
            data.update({"type": "xml", "confidence": 0.9})
            return data
        if "json" in rm:
            data.update({"type": "json", "confidence": 0.9})
            return data

    return data
