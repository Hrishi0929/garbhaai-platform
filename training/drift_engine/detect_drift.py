#!/usr/bin/env python3
"""
Phase 8 drift engine, part 2 (runbook Step 16: "Image Data Drift
Detection"). Meant to run on a schedule (cron -- see
training/drift_engine/README.md for why GitHub Actions doesn't work for a
job that needs localhost Postgres/Redis/Feast on this Mac).

Pipeline, each run:
  1. Pull a sample of recent live embeddings out of Feast's online store --
     via the RECENT_IDS_KEY Redis list services/inference/main.py maintains
     (see that file's _push_to_feast()) and Feast's own
     get_online_features() point-lookup.
  2. Download the training-set baseline that compute_baseline.py logged as
     an MLflow artifact on the current production run.
  3. Compare the two distributions two ways, because they catch different
     shapes of drift:
       - PSI (Population Stability Index), per embedding dimension against
         compute_baseline.py's decile buckets, averaged into one number.
         Catches a shift in any individual dimension's marginal
         distribution.
       - MMD (Maximum Mean Discrepancy, RBF-kernel two-sample test) against
         the raw baseline embeddings, with a permutation test for a
         p-value rather than a hand-picked threshold. Catches a
         joint-distribution shift that could leave every single marginal
         looking unremarkable (a correlation/rotation shift in embedding
         space) -- the kind of drift PSI alone would miss.
  4. Logs the result to a `drift_results` Postgres table (same database
     services/inference/main.py already uses).
  5. If the last SUSTAINED_DRIFT_WINDOW consecutive runs (this one
     included) were ALL flagged, publishes one alert message to the
     `garbhaai:drift_alerts` Redis pub/sub channel on the same Redis
     container Feast's online store already runs on. Requiring several
     consecutive flagged runs (not one) is deliberate: a single noisy
     run on a small live sample is expected sometimes and shouldn't page
     anyone. The runbook is explicit that the actual retrain trigger
     should stay a manual step a human acts on until the rest of this
     pipeline has proven stable -- this script never calls
     training_engine/train.py itself, it only alerts.

Usage:
  cd infra/local && ./bootstrap.sh          # mlflow, postgres, rustfs, redis
  cd ../../training/drift_engine
  pip install -r requirements.txt
  python detect_drift.py
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import psycopg2
import redis
from feast import Entity, FeatureStore
from feast.repo_config import RepoConfig
from feast.value_type import ValueType
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_REGISTRY_NAME = "garbhaai-day-classifier"
PRODUCTION_ALIAS = "production"

# Same local-dev-only RustFS creds as every other script in training/ --
# see compute_baseline.py's comment. Needed here to download the baseline
# artifact (client.download_artifacts), not to upload anything.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "garbhaai")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "garbhaai_local_dev")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

# --- Feast (reading, not writing) -----------------------------------------
# Minimal re-declaration of training/feature_repo/features.py's entity +
# view -- same self-contained-per-process convention as
# services/inference/feast_schema.py (see that file's docstring). This
# process only ever calls get_online_features(), never
# write_to_online_store(), but FeatureStore still needs the view
# registered in ITS OWN registry file to resolve "image_features:embedding"
# feature references -- hence the store.apply() in get_feast_store() below,
# even though semantically this is a read-only job.
FEAST_REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "localhost")
FEAST_REDIS_PORT = os.environ.get("FEAST_REDIS_PORT", "6379")
FEAST_PROJECT = "garbhaai"
FEAST_REGISTRY_PATH = os.environ.get(
    "FEAST_REGISTRY_PATH", os.path.join(os.path.dirname(__file__), "feast_registry.db")
)
RECENT_IDS_KEY = "garbhaai:recent_image_ids"

embryo_image = Entity(
    name="image_id",
    value_type=ValueType.STRING,
    description="Unique identifier for a single embryo image (compute_image_hash()'s output).",
)

OFFLINE_SOURCE_PATH = "feast_offline_placeholder.parquet"

DRIFT_ALERT_CHANNEL = "garbhaai:drift_alerts"

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "garbhaai")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "garbhaai")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "garbhaai_local_dev")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS drift_results (
    id SERIAL PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL,
    production_model_version TEXT,
    production_run_id TEXT,
    n_live_samples INTEGER NOT NULL,
    n_baseline_samples INTEGER,
    skipped BOOLEAN NOT NULL,
    skip_reason TEXT,
    mean_psi REAL,
    max_psi REAL,
    psi_flagged BOOLEAN,
    mmd_squared REAL,
    mmd_p_value REAL,
    mmd_flagged BOOLEAN,
    drift_flagged BOOLEAN,
    sustained_alert_fired BOOLEAN NOT NULL DEFAULT FALSE
);
"""

# Live-sample / statistics knobs. Overridable via env var for testing
# against a thin live stream without waiting for real traffic.
LIVE_SAMPLE_SIZE = int(os.environ.get("DRIFT_LIVE_SAMPLE_SIZE", "50"))
MIN_LIVE_SAMPLES = int(os.environ.get("DRIFT_MIN_LIVE_SAMPLES", "10"))
N_PERMUTATIONS = int(os.environ.get("DRIFT_N_PERMUTATIONS", "200"))
# PSI > 0.25 is the standard industry rule of thumb for "significant
# population shift" (0.1-0.25 is "moderate shift, investigate"); applied
# here to the mean across all 512 dimensions rather than any single one.
PSI_THRESHOLD = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.25"))
MMD_ALPHA = float(os.environ.get("DRIFT_MMD_ALPHA", "0.05"))
# How many consecutive flagged runs (this one included) before alerting --
# see module docstring for why this isn't 1.
SUSTAINED_DRIFT_WINDOW = int(os.environ.get("DRIFT_SUSTAINED_WINDOW", "3"))


def get_feast_store() -> FeatureStore:
    import pandas as pd

    offline_path = os.path.join(os.path.dirname(__file__), OFFLINE_SOURCE_PATH)
    if not os.path.exists(offline_path):
        pd.DataFrame([{
            "image_id": "unused",
            "event_timestamp": datetime.now(timezone.utc),
            "day": 0,
            "model_confidence": 0.0,
            "embedding": [0.0] * 512,
        }]).to_parquet(offline_path)

    config = RepoConfig(
        project=FEAST_PROJECT,
        provider="local",
        registry=FEAST_REGISTRY_PATH,
        online_store={"type": "redis", "connection_string": f"{FEAST_REDIS_HOST}:{FEAST_REDIS_PORT}"},
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
    )
    store = FeatureStore(config=config)

    # Imported here (not module-level) only to keep this function the one
    # place that needs the full view definition -- see feature_repo's
    # features.py for the canonical copy this must stay in sync with.
    from feast import FeatureView, Field, FileSource
    from feast.types import Array, Float32, Int64
    from datetime import timedelta

    image_features_source = FileSource(path=OFFLINE_SOURCE_PATH, timestamp_field="event_timestamp")
    image_features_view = FeatureView(
        name="image_features",
        entities=[embryo_image],
        ttl=timedelta(days=365),
        schema=[
            Field(name="day", dtype=Int64),
            Field(name="model_confidence", dtype=Float32),
            Field(name="embedding", dtype=Array(Float32)),
        ],
        online=True,
        source=image_features_source,
    )
    store.apply([embryo_image, image_features_view])
    return store


def get_raw_redis() -> "redis.Redis":
    return redis.Redis(host=FEAST_REDIS_HOST, port=int(FEAST_REDIS_PORT), decode_responses=True)


def fetch_live_embeddings() -> tuple:
    """Returns (embeddings array (n, 512) or None, n_candidates_checked).
    Reads the most recent LIVE_SAMPLE_SIZE image_hashes off the Redis
    recency list, then looks each one up in Feast's online store --- some
    may come back empty (TTL'd out, or written then the key evicted) so
    the returned array can be smaller than LIVE_SAMPLE_SIZE."""
    r = get_raw_redis()
    recent_ids = r.lrange(RECENT_IDS_KEY, 0, LIVE_SAMPLE_SIZE - 1)
    if not recent_ids:
        return None, 0

    store = get_feast_store()
    result = store.get_online_features(
        features=["image_features:embedding"],
        entity_rows=[{"image_id": image_id} for image_id in recent_ids],
    ).to_dict()

    embeddings = [e for e in result["embedding"] if e is not None]
    if not embeddings:
        return None, len(recent_ids)
    return np.asarray(embeddings, dtype=np.float32), len(recent_ids)


def get_production_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_REGISTRY_NAME, PRODUCTION_ALIAS)
    except Exception:
        return None


def load_baseline(client: MlflowClient, run_id: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        embeddings_path = client.download_artifacts(run_id, "drift_baseline/baseline_embeddings.npz", tmp)
        summary_path = client.download_artifacts(run_id, "drift_baseline/baseline_summary.json", tmp)
        npz = np.load(embeddings_path)
        baseline = {
            "embeddings": npz["embeddings"],
            "bucket_edges": npz["bucket_edges"],
            "bin_proportions": npz["bin_proportions"],
        }
        baseline["summary"] = json.loads(Path(summary_path).read_text())
    return baseline


def compute_psi(live: np.ndarray, bucket_edges: np.ndarray, baseline_bin_proportions: np.ndarray, epsilon: float) -> tuple:
    """Buckets `live` with the SAME edges compute_baseline.py computed from
    the training set, then compares per-bucket proportions dimension by
    dimension. Returns (mean_psi_across_dims, max_psi_across_dims)."""
    n_buckets, n_dims = baseline_bin_proportions.shape
    n_live = live.shape[0]
    live_bin_counts = np.zeros((n_buckets, n_dims), dtype=np.float64)
    for dim in range(n_dims):
        bin_idx = np.digitize(live[:, dim], bucket_edges[1:-1, dim], right=False)
        bin_idx = np.clip(bin_idx, 0, n_buckets - 1)  # a live value outside the baseline's observed range
        for b in range(n_buckets):
            live_bin_counts[b, dim] = np.sum(bin_idx == b)
    live_bin_proportions = live_bin_counts / n_live

    base_p = np.clip(baseline_bin_proportions, epsilon, None)
    live_p = np.clip(live_bin_proportions, epsilon, None)
    psi_per_dim = np.sum((live_p - base_p) * np.log(live_p / base_p), axis=0)  # (n_dims,)
    return float(np.mean(psi_per_dim)), float(np.max(psi_per_dim))


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = np.sum(A ** 2, axis=1)[:, None]
    B2 = np.sum(B ** 2, axis=1)[None, :]
    return np.maximum(A2 + B2 - 2 * A @ B.T, 0.0)


def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, sigma2: float) -> float:
    def kernel(A, B):
        return np.exp(-_sq_dists(A, B) / sigma2)

    m, n = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = kernel(X, X), kernel(Y, Y), kernel(X, Y)
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.sum() / (m * n)
    return float(sum_xx + sum_yy - 2 * sum_xy)


def compute_mmd(live: np.ndarray, baseline_embeddings: np.ndarray, n_permutations: int, seed: int = 0) -> tuple:
    """RBF-kernel MMD^2 between the live sample and the training-set
    baseline, with a permutation test for a p-value (verified empirically
    against synthetic same-distribution vs. shifted-distribution samples
    before being used here -- see this phase's working notes). Median
    heuristic for the kernel bandwidth, standard practice when there's no
    principled alternative. Returns (mmd_squared, p_value)."""
    rng = np.random.default_rng(seed)
    combined = np.concatenate([live, baseline_embeddings], axis=0)
    sq = _sq_dists(combined, combined)
    iu = np.triu_indices_from(sq, k=1)
    median_sq_dist = float(np.median(sq[iu]))
    sigma2 = median_sq_dist if median_sq_dist > 0 else 1.0

    observed = _rbf_mmd2(live, baseline_embeddings, sigma2)

    m = live.shape[0]
    count_ge = 0
    for _ in range(n_permutations):
        idx = rng.permutation(combined.shape[0])
        Xp, Yp = combined[idx[:m]], combined[idx[m:]]
        stat = _rbf_mmd2(Xp, Yp, sigma2)
        if stat >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)  # +1/+1: never report p=0, matches standard permutation-test convention
    return observed, p_value


def get_db_conn():
    conn = psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    )
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def log_result(conn, row: dict) -> int:
    columns = list(row.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO drift_results ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id",
            [row[c] for c in columns],
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def check_sustained_drift(conn) -> bool:
    """True iff the last SUSTAINED_DRIFT_WINDOW non-skipped runs
    (including the one just logged) were ALL drift_flagged. A single
    noisy run shouldn't trigger an alert -- see module docstring."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT drift_flagged FROM drift_results WHERE skipped = FALSE "
            "ORDER BY checked_at DESC LIMIT %s",
            (SUSTAINED_DRIFT_WINDOW,),
        )
        rows = [r[0] for r in cur.fetchall()]
    return len(rows) == SUSTAINED_DRIFT_WINDOW and all(rows)


def publish_alert(payload: dict) -> None:
    r = get_raw_redis()
    r.publish(DRIFT_ALERT_CHANNEL, json.dumps(payload))


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    conn = get_db_conn()
    checked_at = datetime.now(timezone.utc)

    production = get_production_version(client)
    base_row = {
        "checked_at": checked_at,
        "production_model_version": production.version if production else None,
        "production_run_id": production.run_id if production else None,
        "n_baseline_samples": None,
        "mean_psi": None, "max_psi": None, "psi_flagged": None,
        "mmd_squared": None, "mmd_p_value": None, "mmd_flagged": None,
        "drift_flagged": None, "sustained_alert_fired": False,
    }

    if production is None:
        print(f"SKIPPED: no version holds the '{PRODUCTION_ALIAS}' alias yet.")
        log_result(conn, {**base_row, "n_live_samples": 0, "skipped": True,
                           "skip_reason": "no production model registered"})
        conn.close()
        return

    live, n_candidates = fetch_live_embeddings()
    if live is None or live.shape[0] < MIN_LIVE_SAMPLES:
        n_found = 0 if live is None else live.shape[0]
        reason = f"only {n_found} live embeddings available (need {MIN_LIVE_SAMPLES}), out of {n_candidates} recent ids checked"
        print(f"SKIPPED: {reason}")
        log_result(conn, {**base_row, "n_live_samples": n_found, "skipped": True, "skip_reason": reason})
        conn.close()
        return

    print(f"Loaded {live.shape[0]} live embeddings (checked {n_candidates} recent ids)")
    print(f"Downloading baseline from production run {production.run_id}...")
    baseline = load_baseline(client, production.run_id)
    n_baseline = baseline["embeddings"].shape[0]
    print(f"Baseline has {n_baseline} training-set embeddings")

    summary = baseline["summary"]
    epsilon = summary.get("psi_epsilon", 1e-4)
    mean_psi, max_psi = compute_psi(live, baseline["bucket_edges"], baseline["bin_proportions"], epsilon)
    psi_flagged = mean_psi > PSI_THRESHOLD

    mmd_squared, mmd_p_value = compute_mmd(live, baseline["embeddings"], N_PERMUTATIONS)
    mmd_flagged = mmd_p_value < MMD_ALPHA

    drift_flagged = bool(psi_flagged or mmd_flagged)

    print(f"PSI: mean={mean_psi:.4f} max={max_psi:.4f} (threshold {PSI_THRESHOLD}) -> flagged={psi_flagged}")
    print(f"MMD: mmd^2={mmd_squared:.6f} p_value={mmd_p_value:.4f} (alpha {MMD_ALPHA}) -> flagged={mmd_flagged}")
    print(f"Overall drift_flagged = {drift_flagged}")

    row_id = log_result(conn, {
        **base_row,
        "n_live_samples": int(live.shape[0]),
        "n_baseline_samples": n_baseline,
        "skipped": False,
        "skip_reason": None,
        "mean_psi": mean_psi, "max_psi": max_psi, "psi_flagged": psi_flagged,
        "mmd_squared": mmd_squared, "mmd_p_value": mmd_p_value, "mmd_flagged": mmd_flagged,
        "drift_flagged": drift_flagged,
    })

    sustained = check_sustained_drift(conn) if drift_flagged else False
    if sustained:
        payload = {
            "event": "sustained_drift",
            "checked_at": checked_at.isoformat(),
            "window": SUSTAINED_DRIFT_WINDOW,
            "production_model_version": production.version,
            "production_run_id": production.run_id,
            "mean_psi": mean_psi,
            "mmd_p_value": mmd_p_value,
            "action_required": (
                "A human should review this before retraining -- this job deliberately "
                "never triggers training_engine/train.py itself."
            ),
        }
        publish_alert(payload)
        with conn.cursor() as cur:
            cur.execute("UPDATE drift_results SET sustained_alert_fired = TRUE WHERE id = %s", (row_id,))
        conn.commit()
        print(f"\nALERT: {SUSTAINED_DRIFT_WINDOW} consecutive flagged runs -- published to '{DRIFT_ALERT_CHANNEL}'.")
        print("No retrain was triggered automatically -- see module docstring.")

    conn.close()


if __name__ == "__main__":
    main()
