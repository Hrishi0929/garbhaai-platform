"""
Inference service: the day-conditioned classifier head, the doctor
review flow, and the labeled-data store.

Ported from garbhaai-embryo-grading/src/grading_core.py (the
classifier-head half of DayConditionedClassifier, and image_hash) and
src/records_store.py (the accept/override review flow, and the
image-hash-based "never contradict a decision on an image we've already
seen" cache) -- but records_store's local JSON file becomes the
`grade_records` Postgres table here, and the model itself is loaded
from a checkpoint bundled into this service's image (model/unified_day3_day4_classifier.pt)
rather than a local path. Phase 6 (MLflow) is expected to replace this
bundled-file loading with a real model registry pull -- this is a
deliberate placeholder, not the final design.

Does NOT reimplement the ResNet18 backbone -- the checkpoint only ever
held the classifier head (Linear(514, 3): 512 backbone features + a
2-dim day one-hot -> 3 grade logits; verified directly from the
checkpoint's own keys and confirmed against src/train_unified.py's
torch.save() call). This service calls the feature-extraction service
over HTTP for the 512-dim vector instead, which is the correct way to
reuse work across services -- through an API, not a shared library.

Day 5 is out of scope: the model was only ever trained on the pooled
Day3+Day4 dataset (a decision made earlier in the project, and one the
Phase 3 data-quality gate independently corroborated by finding Day5
uses a different grading taxonomy). /grade rejects day=5 explicitly
rather than silently producing a meaningless prediction for it.
"""
import hashlib
import io
import json
import os
from datetime import datetime, timezone

import httpx
import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="garbhaai-inference")

FEATURE_EXTRACTION_URL = os.environ.get("FEATURE_EXTRACTION_URL", "http://localhost:8004")
MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    os.path.join(os.path.dirname(__file__), "model", "unified_day3_day4_classifier.pt"),
)

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "garbhaai")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "garbhaai")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "garbhaai_local_dev")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS grade_records (
    image_hash TEXT PRIMARY KEY,
    day INTEGER NOT NULL,
    model_grade TEXT NOT NULL,
    model_confidence REAL NOT NULL,
    model_probabilities JSONB NOT NULL,
    final_grade TEXT NOT NULL,
    doctor_overridden BOOLEAN NOT NULL,
    patient_id TEXT,
    reviewed_at TIMESTAMPTZ NOT NULL
);
"""

_checkpoint_cache = {}


def compute_image_hash(image_bytes: bytes) -> str:
    """Matches ingestion's compute_image_hash() / the original
    grading_core.image_hash() -- md5, first 8 hex chars."""
    return hashlib.md5(image_bytes).hexdigest()[:8]


def load_checkpoint() -> dict:
    if "checkpoint" not in _checkpoint_cache:
        ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _checkpoint_cache["checkpoint"] = ckpt
    return _checkpoint_cache["checkpoint"]


def build_classifier_head(checkpoint: dict) -> nn.Module:
    state_dict = checkpoint["classifier_state_dict"]
    in_features, out_features = state_dict["weight"].shape[1], state_dict["weight"].shape[0]
    head = nn.Linear(in_features, out_features)
    head.load_state_dict(state_dict)
    head.eval()
    return head


def day_key(day: int) -> str:
    key = f"day{day}"
    checkpoint = load_checkpoint()
    if key not in checkpoint["day_onehot"]:
        supported = ", ".join(sorted(k.replace("day", "") for k in checkpoint["day_onehot"]))
        raise HTTPException(
            status_code=422,
            detail=f"day={day} is not supported by this model (trained on days: {supported}).",
        )
    return key


def predict_grade(features: list, day: int) -> dict:
    checkpoint = load_checkpoint()
    head = build_classifier_head(checkpoint)
    onehot = checkpoint["day_onehot"][day_key(day)]
    class_names = checkpoint["class_names"]

    feature_vec = np.asarray(features, dtype=np.float32)
    onehot_vec = np.asarray(onehot, dtype=np.float32)
    vector = np.concatenate([feature_vec, onehot_vec])
    with torch.no_grad():
        logits = head(torch.from_numpy(vector).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    best_idx = int(np.argmax(probs))
    return {
        "grade": class_names[best_idx],
        "confidence": float(probs[best_idx]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }


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


def fetch_existing_record(db, image_hash: str):
    with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM grade_records WHERE image_hash = %s", (image_hash,))
        return cur.fetchone()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/grade")
async def grade(
    day: int = Form(...),
    file: UploadFile = File(...),
    db=Depends(get_db_conn),
):
    image_bytes = await file.read()
    # validate the file is at least a readable image before doing any
    # network calls or model work on it
    Image.open(io.BytesIO(image_bytes)).verify()
    image_hash = compute_image_hash(image_bytes)

    existing = fetch_existing_record(db, image_hash)
    if existing is not None:
        return {
            "source": "cached_review",
            "image_hash": image_hash,
            "grade": existing["final_grade"],
            "confidence": existing["model_confidence"],
            "probabilities": existing["model_probabilities"],
            "doctor_overridden": existing["doctor_overridden"],
        }

    day_key(day)  # raises 422 for an unsupported day before calling out to feature-extraction

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{FEATURE_EXTRACTION_URL}/extract-features",
            files={"file": (file.filename or "image.jpg", image_bytes, "image/jpeg")},
        )
    resp.raise_for_status()
    features = resp.json()["features"]

    prediction = predict_grade(features, day)
    return {
        "source": "model",
        "image_hash": image_hash,
        **prediction,
    }


@app.post("/review")
async def review(
    image_hash: str = Form(...),
    day: int = Form(...),
    model_grade: str = Form(...),
    model_confidence: float = Form(...),
    model_probabilities: str = Form(...),  # JSON-encoded dict, form fields are flat strings
    final_grade: str = Form(...),
    doctor_overridden: bool = Form(...),
    patient_id: str = Form(None),
    db=Depends(get_db_conn),
):
    probabilities = json.loads(model_probabilities)
    reviewed_at = datetime.now(timezone.utc)

    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO grade_records (
                image_hash, day, model_grade, model_confidence, model_probabilities,
                final_grade, doctor_overridden, patient_id, reviewed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_hash) DO UPDATE SET
                final_grade = EXCLUDED.final_grade,
                doctor_overridden = EXCLUDED.doctor_overridden,
                patient_id = EXCLUDED.patient_id,
                reviewed_at = EXCLUDED.reviewed_at
            """,
            (
                image_hash, day, model_grade, model_confidence, json.dumps(probabilities),
                final_grade, doctor_overridden, patient_id, reviewed_at,
            ),
        )
    db.commit()

    return {
        "image_hash": image_hash,
        "final_grade": final_grade,
        "doctor_overridden": doctor_overridden,
    }
