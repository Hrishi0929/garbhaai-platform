"""
Ingestion service: receives an uploaded embryo image (patient_id + day +
file), stores the bytes in object storage, and records the ingestion in
Postgres.

Ported from garbhaai-embryo-grading/src/records_store.py, but the
storage target changes: the old app wrote to a local JSON file
(data/clinical_records/records.json) and local disk
(data/clinical_records/images/). This service writes to RustFS (the
Phase 2 S3-compatible object store -- swaps for Google Cloud Storage at
real-deployment time, see infra/local/docker-compose.yml) and Postgres
(also stood up in Phase 2), behind the get_storage_client()/get_db_conn()
dependencies below, so that swap stays a one-place change.

The original app's image-hash-based re-grading cache (same image bytes
=> same answer, regardless of patient_id, because the model must never
contradict itself on an image it's already seen with only ~45 training
examples) is NOT reimplemented here -- that decision belongs with the
Phase 5 inference service, which is what actually decides whether to
reuse a cached grade. This service's job stops at: store the image
once per unique hash, record every ingestion event (even of an
already-seen hash, since the same image can legitimately be
re-submitted under a different patient_id), and report back whether
the hash was already known.
"""
import hashlib
import io
import os
import uuid
from datetime import datetime, timezone

import boto3
import psycopg2
from fastapi import Depends, FastAPI, File, Form, UploadFile

app = FastAPI(title="garbhaai-ingestion")

RUSTFS_ENDPOINT = os.environ.get("RUSTFS_ENDPOINT", "http://localhost:9000")
RUSTFS_ACCESS_KEY = os.environ.get("RUSTFS_ACCESS_KEY", "garbhaai")
RUSTFS_SECRET_KEY = os.environ.get("RUSTFS_SECRET_KEY", "garbhaai_local_dev")
RUSTFS_BUCKET = os.environ.get("RUSTFS_BUCKET", "garbhaai-images")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "garbhaai")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "garbhaai")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "garbhaai_local_dev")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS images (
    image_id UUID PRIMARY KEY,
    image_hash TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    storage_key TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_hash ON images (image_hash);
"""


def compute_image_hash(image_bytes: bytes) -> str:
    """Matches grading_core.py's image_hash(): md5, first 8 hex chars.
    Kept short since it's used as an object-storage key component, not
    for cryptographic purposes."""
    return hashlib.md5(image_bytes).hexdigest()[:8]


def build_storage_key(image_hash: str, day: int) -> str:
    return f"day{day}/{image_hash}.jpg"


def get_storage_client():
    return boto3.client(
        "s3",
        endpoint_url=RUSTFS_ENDPOINT,
        aws_access_key_id=RUSTFS_ACCESS_KEY,
        aws_secret_access_key=RUSTFS_SECRET_KEY,
    )


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    patient_id: str = Form(...),
    day: int = Form(...),
    file: UploadFile = File(...),
    storage=Depends(get_storage_client),
    db=Depends(get_db_conn),
):
    image_bytes = await file.read()
    image_hash = compute_image_hash(image_bytes)
    storage_key = build_storage_key(image_hash, day)

    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM images WHERE image_hash = %s LIMIT 1", (image_hash,))
        already_seen = cur.fetchone() is not None

    if not already_seen:
        storage.put_object(
            Bucket=RUSTFS_BUCKET,
            Key=storage_key,
            Body=io.BytesIO(image_bytes),
            ContentLength=len(image_bytes),
        )

    image_id = str(uuid.uuid4())
    ingested_at = datetime.now(timezone.utc)
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO images (image_id, image_hash, patient_id, day, storage_key, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (image_id, image_hash, patient_id, day, storage_key, ingested_at),
        )
    db.commit()

    return {
        "image_id": image_id,
        "image_hash": image_hash,
        "storage_key": storage_key,
        "already_seen": already_seen,
    }
