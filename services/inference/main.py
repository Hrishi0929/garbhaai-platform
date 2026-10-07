"""
Inference service: the day-conditioned classifier head, the doctor
review flow, and the labeled-data store.

Ported from garbhaai-embryo-grading/src/grading_core.py (the
classifier-head half of DayConditionedClassifier, and image_hash) and
src/records_store.py (the accept/override review flow, and the
image-hash-based "never contradict a decision on an image we've already
seen" cache) -- but records_store's local JSON file becomes the
`grade_records` Postgres table here.

Phase 7: the model is now pulled from the MLflow registry at startup --
specifically, whichever version currently holds the "production" alias
of the garbhaai-day-classifier registered model (see
training/promote_engine/promote.py, which is the only thing that ever
moves that alias). The bundled checkpoint
(model/unified_day3_day4_classifier.pt) is kept as a local-dev fallback
for when MLflow is unreachable (offline laptop work, a fresh clone
before any training run has happened) -- not the primary path anymore.
load_checkpoint() tries the registry first and only falls back on
failure, logging loudly either way so it's never ambiguous from the
logs (or from GET /health, which reports the source and version) which
one a given pod is actually serving.

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

Phase 8: every live (non-cached) /grade prediction now also stores its
512-dim feature vector in the Postgres `image_features` table (see
_push_features()). That table is the platform's feature store: one row per
image (keyed by image_hash) holding the day, the model's confidence, the
embedding and a timestamp. It replaces the earlier Feast + Redis setup --
Postgres is already part of the deployment, so there is one fewer moving
part to run, and the same table serves live reads (drift detection asks
for the most recent rows) and offline reads (training can read all of
it). training/drift_engine/detect_drift.py is what reads it back out.
Writing is best-effort and never allowed to fail a grading request: a
database hiccup logs a warning and the prediction still returns normally.
"""
import hashlib
import io
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import httpx
import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

logger = logging.getLogger("garbhaai.inference")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="garbhaai-inference")

FEATURE_EXTRACTION_URL = os.environ.get("FEATURE_EXTRACTION_URL", "http://localhost:8004")
MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    os.path.join(os.path.dirname(__file__), "model", "unified_day3_day4_classifier.pt"),
)

# --- MLflow registry pull (Phase 7) ---------------------------------------
# Same registered-model name train.py writes to and promote.py promotes
# within. "production" is an alias (mlflow.set_registered_model_alias), not
# a legacy numbered stage -- MLflow deprecated
# transition_model_version_stage back in 2.9.0 (confirmed against the
# actual installed mlflow-skinny 3.16.1 source: the method is still present
# but carries a @deprecated decorator), so promote.py and this file both
# use the alias API, which is the maintained one.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_REGISTRY_NAME = "garbhaai-day-classifier"
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")

# Same local-dev-only RustFS credentials as train.py/evaluate.py/promote.py
# -- see train.py's comment for the full explanation of why this is needed
# outside Docker at all. Inside the kind cluster this is instead set for
# real via the Rollout's env (infra/k8s/inference/rollout.yaml), pointed at
# host.docker.internal since RustFS still runs in docker-compose on the
# Mac host, not inside the cluster.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "garbhaai")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "garbhaai_local_dev")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

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

-- Feature store (Phase 8): one row per live-graded image. `embedding` is the
-- 512-dim backbone vector. The timestamp index serves "most recent N rows",
-- which is what drift detection asks for.
CREATE TABLE IF NOT EXISTS image_features (
    image_hash TEXT PRIMARY KEY,
    day INTEGER NOT NULL,
    model_confidence REAL NOT NULL,
    embedding REAL[] NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_image_features_ts ON image_features (event_timestamp DESC);
"""

UPSERT_FEATURES_SQL = """
INSERT INTO image_features (image_hash, day, model_confidence, embedding, event_timestamp)
VALUES (%s, %s, %s, %s::real[], %s)
ON CONFLICT (image_hash) DO UPDATE SET
    day = EXCLUDED.day,
    model_confidence = EXCLUDED.model_confidence,
    embedding = EXCLUDED.embedding,
    event_timestamp = EXCLUDED.event_timestamp
"""

_checkpoint_cache = {}


def _push_features(db, image_hash: str, day: int, confidence: float, features: list) -> None:
    """Best-effort: a database problem here must never break a grading
    request, which is this service's actual job. Logs a warning and moves
    on -- detect_drift.py simply sees fewer live samples that run, same as
    any other monitoring gap."""
    try:
        with db.cursor() as cur:
            cur.execute(
                UPSERT_FEATURES_SQL,
                (image_hash, day, confidence, [float(x) for x in features],
                 datetime.now(timezone.utc)),
            )
        db.commit()
    except Exception as exc:
        logger.warning("could not store features for image %s (continuing): %s", image_hash, exc)
        try:
            db.rollback()
        except Exception:
            pass


def compute_image_hash(image_bytes: bytes) -> str:
    """Matches ingestion's compute_image_hash() / the original
    grading_core.image_hash() -- md5, first 8 hex chars."""
    return hashlib.md5(image_bytes).hexdigest()[:8]


def _load_checkpoint_from_registry() -> tuple:
    """Downloads whichever version currently holds the MODEL_ALIAS alias
    of MODEL_REGISTRY_NAME and loads it. Raises on any failure (unreachable
    MLflow, no version holds that alias yet, a malformed artifact) --
    load_checkpoint() is what decides whether to fall back, this function
    just tries the real thing and reports exactly what it got."""
    import mlflow  # imported lazily so a from-bundle-only dev environment
    from mlflow.tracking import MlflowClient  # never needs mlflow installed at all

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, MODEL_ALIAS)

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = [a.path for a in client.list_artifacts(mv.run_id)]
        checkpoint_name = next((a for a in artifacts if a.endswith(".pt")), None)
        if checkpoint_name is None:
            raise RuntimeError(
                f"registered version {mv.version} (run {mv.run_id}) has no .pt artifact "
                f"(artifacts found: {artifacts})"
            )
        local_path = client.download_artifacts(mv.run_id, checkpoint_name, tmp)
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)

    source_info = {
        "source": "registry",
        "registered_model": MODEL_REGISTRY_NAME,
        "alias": MODEL_ALIAS,
        "version": mv.version,
        "run_id": mv.run_id,
    }
    return ckpt, source_info


def _load_checkpoint_from_bundle() -> tuple:
    ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    source_info = {"source": "bundled_fallback", "path": MODEL_CHECKPOINT_PATH}
    return ckpt, source_info


def load_checkpoint() -> dict:
    if "checkpoint" not in _checkpoint_cache:
        try:
            ckpt, source_info = _load_checkpoint_from_registry()
            logger.info(
                "loaded model from MLflow registry: %s version %s (alias=%s, run=%s)",
                MODEL_REGISTRY_NAME, source_info["version"], MODEL_ALIAS, source_info["run_id"],
            )
        except Exception as exc:
            logger.warning(
                "could not load model from MLflow registry (%s) -- falling back to the "
                "bundled checkpoint at %s. This is expected in local dev before any "
                "training run exists; it should NOT happen in a deployed environment "
                "once Phase 6/7 have run.",
                exc, MODEL_CHECKPOINT_PATH,
            )
            ckpt, source_info = _load_checkpoint_from_bundle()
        _checkpoint_cache["checkpoint"] = ckpt
        _checkpoint_cache["source_info"] = source_info
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
    # Loads the model if this is the first request to hit this pod (same
    # lazy-cache path /grade uses) specifically so /health can report which
    # model is actually serving -- registry version vs. the bundled
    # fallback -- without needing a /grade request first. This is the
    # signal Phase 7's canary verification relies on: curling a specific
    # pod's /health after a promotion confirms it picked up the new
    # version rather than assuming so from the Rollout status alone.
    load_checkpoint()
    return {"status": "ok", "model": _checkpoint_cache.get("source_info")}


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

    # Only live predictions are new information for drift detection -- a
    # cached_review response above returns early and never reaches here,
    # which is correct: it's the same image seen before, not a fresh
    # sample of what the model is being asked to grade right now.
    _push_features(db, image_hash, day, prediction["confidence"], features)

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
