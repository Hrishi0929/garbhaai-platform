#!/usr/bin/env python3
"""
Phase 8 drift engine, part 1: computes the training-set baseline that
detect_drift.py (part 2) compares live Feast-pushed embeddings against.

Walks the full Day3+Day4 training dataset -- the exact same folder walk and
feature-extraction call as training/training_engine/train.py's load_stage()/
extract_features(), duplicated here rather than imported across the
training_engine/drift_engine boundary (each subfolder under training/ is
self-contained in this repo -- see services/inference/feast_schema.py's
docstring for the same reasoning applied to a service boundary). Keep the
two walks in sync by hand if the dataset layout changes.

What "baseline" means here, concretely: for each of the 512 embedding
dimensions independently, the decile bucket edges of the training-set
values in that dimension, plus what fraction of training images fell into
each bucket. detect_drift.py buckets a live sample the same way and
compares bucket proportions per dimension -- that comparison is the
Population Stability Index (PSI), averaged across dimensions into one
number. The raw embedding matrix is also kept (not just the summary), so
detect_drift.py can additionally run Maximum Mean Discrepancy (MMD, a
kernel two-sample test) directly against real baseline samples rather than
only the bucketed summary -- PSI and MMD catch different shapes of drift
(per-dimension marginal shift vs. a joint-distribution shift that keeps
every marginal looking fine), which is why the runbook (Phase 8, Step 16)
asks for both.

The baseline is logged as an MLflow artifact on whichever run currently
holds the "production" alias of garbhaai-day-classifier -- not a new run --
so it stays tied to the model version it was computed against. If that
model is later replaced, re-running this script attaches a fresh baseline
to the new production run; detect_drift.py always reads the baseline off
of whatever run is current production, so there's no separate "which
baseline is active" bookkeeping to maintain.

Usage:
  cd infra/local && ./bootstrap.sh          # mlflow, postgres, rustfs, redis
  cd ../../services && docker compose up -d  # feature-extraction, etc.
  cd ../training/drift_engine
  pip install -r requirements.txt
  python compute_baseline.py
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import requests
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]  # garbhaai-platform/
DATASET_ROOT = Path(os.environ.get("GARBHAAI_DATASET_ROOT", REPO_ROOT.parent / "Dataset"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEATURE_EXTRACTION_URL = os.environ.get("FEATURE_EXTRACTION_URL", "http://localhost:8004")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_REGISTRY_NAME = "garbhaai-day-classifier"
PRODUCTION_ALIAS = "production"

# mlflow's S3 artifact repo imports boto3 lazily, only inside
# log_artifact() -- same gap documented in training_engine/train.py and
# eval_engine/evaluate.py. Same local-dev-only RustFS creds
# infra/local/docker-compose.yml uses. setdefault so a real env var
# (e.g. a different host) still wins.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "garbhaai")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "garbhaai_local_dev")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["A", "B", "C"]
STAGES = ["day3", "day4"]  # matches train.py -- day5 excluded, different grading taxonomy
DAY_DIR_NAME = {"day3": "Day 3 Dataset", "day4": "Day 4 Dataset"}
EMBEDDING_DIM = 512
N_BUCKETS = 10  # deciles -- enough resolution for PSI at this sample size (~30-45 images)
PSI_EPSILON = 1e-4  # floor for zero-proportion buckets, standard PSI smoothing


def extract_features(image_path: Path) -> list:
    """Identical call to train.py's extract_features() -- same service,
    same endpoint, so the baseline is built from the exact backbone
    production inference and training both use."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{FEATURE_EXTRACTION_URL}/extract-features",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["features"]


def load_all_embeddings() -> tuple:
    """Walks Dataset/Day N Dataset/Grade X/*.jpg for every stage in STAGES
    -- same walk as train.py's load_stage(), minus the day-onehot/label
    bookkeeping train.py needs for classification and this script doesn't
    (drift is tracked on the embedding space alone, independent of day or
    grade)."""
    features, image_ids, stage_labels = [], [], []
    for stage in STAGES:
        stage_dir = DATASET_ROOT / DAY_DIR_NAME[stage]
        if not stage_dir.exists():
            raise SystemExit(f"dataset folder not found: {stage_dir}")

        stage_count = 0
        for grade_dir in sorted(stage_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            # Same "trailing token" parse as train.py's load_stage() --
            # day3 folders are "Grade A"/"Grade B"/"Grade C", day4 folders
            # are "Morula Grade A"/... -- see that function's comment for
            # why a substring strip silently breaks on day4.
            grade = grade_dir.name.strip().split()[-1]
            if grade not in CLASS_NAMES:
                continue
            for f in sorted(grade_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                features.append(extract_features(f))
                image_ids.append(f.stem)
                stage_labels.append(stage)
                stage_count += 1
        print(f"  {stage}: {stage_count} images")

    if not features:
        raise SystemExit(f"no images found under {DATASET_ROOT} -- refusing to compute an empty baseline")

    X = np.asarray(features, dtype=np.float32)
    if X.shape[1] != EMBEDDING_DIM:
        raise SystemExit(
            f"feature-extraction returned {X.shape[1]}-dim vectors, expected {EMBEDDING_DIM} -- "
            f"baseline and live embeddings must be the same shape for drift comparison to mean anything"
        )
    return X, image_ids, stage_labels


def compute_decile_buckets(X: np.ndarray) -> tuple:
    """Per-dimension decile edges (N_BUCKETS+1, 512) and the fraction of
    training samples landing in each bucket (N_BUCKETS, 512). This is the
    PSI reference distribution: detect_drift.py buckets a live sample with
    these same edges and compares its per-bucket proportions against
    baseline_bin_proportions, dimension by dimension."""
    quantile_points = np.linspace(0.0, 1.0, N_BUCKETS + 1)
    bucket_edges = np.quantile(X, quantile_points, axis=0).astype(np.float32)  # (N_BUCKETS+1, 512)

    # Nudge the outer edges so np.digitize never drops a boundary value
    # into a nonexistent 11th bucket.
    bucket_edges[0, :] -= 1e-6
    bucket_edges[-1, :] += 1e-6

    n_images, n_dims = X.shape
    bin_counts = np.zeros((N_BUCKETS, n_dims), dtype=np.float64)
    for dim in range(n_dims):
        # right=False + edges nudged above means digitize returns 1..N_BUCKETS
        bin_idx = np.digitize(X[:, dim], bucket_edges[1:-1, dim], right=False)
        for b in range(N_BUCKETS):
            bin_counts[b, dim] = np.sum(bin_idx == b)
    bin_proportions = (bin_counts / n_images).astype(np.float32)
    return bucket_edges, bin_proportions


def get_production_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_REGISTRY_NAME, PRODUCTION_ALIAS)
    except Exception:
        return None


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Extracting embeddings for {STAGES} from {DATASET_ROOT}...")
    X, image_ids, stage_labels = load_all_embeddings()
    print(f"Pooled: {X.shape[0]} images, {X.shape[1]}-dim embeddings")

    bucket_edges, bin_proportions = compute_decile_buckets(X)

    production = get_production_version(client)
    if production is None:
        raise SystemExit(
            f"No version currently holds the '{PRODUCTION_ALIAS}' alias on '{MODEL_REGISTRY_NAME}' -- "
            f"run training/promote_engine/promote.py at least once before computing a baseline, "
            f"so there's a production run to attach it to."
        )
    print(f"Attaching baseline to production version {production.version} (run {production.run_id})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = OUTPUT_DIR / "baseline_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=X,
        bucket_edges=bucket_edges,
        bin_proportions=bin_proportions,
    )

    summary = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "production_model_name": MODEL_REGISTRY_NAME,
        "production_model_version": production.version,
        "production_run_id": production.run_id,
        "n_images": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "n_buckets": N_BUCKETS,
        "psi_epsilon": PSI_EPSILON,
        "stages_included": STAGES,
        "stage_counts": {s: stage_labels.count(s) for s in STAGES},
        "image_ids": image_ids,
    }
    summary_path = OUTPUT_DIR / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    client.log_artifact(production.run_id, str(embeddings_path), artifact_path="drift_baseline")
    client.log_artifact(production.run_id, str(summary_path), artifact_path="drift_baseline")

    print(f"\nWrote {embeddings_path}")
    print(f"Wrote {summary_path}")
    print(
        f"Logged both as artifacts under drift_baseline/ on run {production.run_id} "
        f"(view at {MLFLOW_TRACKING_URI})"
    )


if __name__ == "__main__":
    main()
