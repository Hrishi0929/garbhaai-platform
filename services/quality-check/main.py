"""
Quality-check service: blur/exposure/integrity check on an uploaded
embryo image, run before it's sent anywhere else in the pipeline.

Ported from garbhaai-embryo-grading/src/grading_core.py's
check_image_quality(), unchanged in logic -- this service just wraps it
behind an HTTP API instead of a direct in-process import, so ingestion,
the UI, or anything else can call it without needing PyTorch installed
(this service has no model dependency at all).

Phase 9 gap-closure: the check result now also gets written to a
quality_checks Postgres row, keyed by the same md5[:8] image_hash
convention ingestion/inference use -- previously this service computed
{ok, warnings} and handed it straight back to the caller with no record
kept anywhere, so there was no audit trail of a quality-check step ever
having run on a given image. The runbook's Phase 9 smoke-test checklist
(item 2) explicitly checks for a row here, separate from and not
covered by Feast (Feast only holds the feature embedding, not this
pass/fail decision).
"""
import hashlib
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import psycopg2
from fastapi import Depends, FastAPI, File, UploadFile
from PIL import Image
from scipy.ndimage import laplace

app = FastAPI(title="garbhaai-quality-check")

# NOT calibrated against this dataset -- there were no known-blurry example
# images to tune against when this was written. If real blurry uploads slip
# through (or good images get flagged), adjust this after seeing a few.
BLUR_VARIANCE_THRESHOLD = 100.0
UNDEREXPOSED_MEAN_THRESHOLD = 20.0   # 0-255 grayscale scale
OVEREXPOSED_MEAN_THRESHOLD = 235.0

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "garbhaai")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "garbhaai")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "garbhaai_local_dev")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS quality_checks (
    image_hash TEXT NOT NULL,
    ok BOOLEAN NOT NULL,
    warnings JSONB NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quality_checks_hash ON quality_checks (image_hash);
"""


def compute_image_hash(image_bytes: bytes) -> str:
    """Matches ingestion's/inference's convention: md5, first 8 hex
    chars -- so a quality_checks row can be correlated back to the same
    image's images/grade_records rows by image_hash."""
    return hashlib.md5(image_bytes).hexdigest()[:8]


def get_db_conn():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        yield conn
    finally:
        conn.close()


def check_image_quality(pil_image: Image.Image) -> dict:
    """Returns {"ok": bool, "warnings": [str, ...]}. "ok" is False only
    for a genuinely unreadable file; blur/exposure are warnings the
    caller (UI) can choose to let the user proceed past, not hard
    failures."""
    warnings = []
    try:
        gray = np.asarray(pil_image.convert("L"), dtype=np.float32)
    except Exception as e:
        return {"ok": False, "warnings": [f"Could not read this image file ({e})."]}

    if gray.size == 0:
        return {"ok": False, "warnings": ["Image appears to be empty/corrupt."]}

    blur_variance = float(np.var(laplace(gray)))
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        warnings.append(
            f"Image looks blurred (sharpness score {blur_variance:.0f}, "
            f"below the {BLUR_VARIANCE_THRESHOLD:.0f} threshold)."
        )

    mean_brightness = float(gray.mean())
    if mean_brightness < UNDEREXPOSED_MEAN_THRESHOLD:
        warnings.append(
            f"Image looks under-exposed / too dark (mean brightness {mean_brightness:.0f})."
        )
    elif mean_brightness > OVEREXPOSED_MEAN_THRESHOLD:
        warnings.append(
            f"Image looks over-exposed / too bright (mean brightness {mean_brightness:.0f})."
        )

    return {"ok": True, "warnings": warnings}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-quality")
async def check_quality(file: UploadFile = File(...), db=Depends(get_db_conn)):
    image_bytes = await file.read()
    image_hash = compute_image_hash(image_bytes)
    pil_image = Image.open(io.BytesIO(image_bytes))
    result = check_image_quality(pil_image)

    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quality_checks (image_hash, ok, warnings, checked_at)
            VALUES (%s, %s, %s, %s)
            """,
            (image_hash, result["ok"], json.dumps(result["warnings"]), datetime.now(timezone.utc)),
        )
    db.commit()

    return result
