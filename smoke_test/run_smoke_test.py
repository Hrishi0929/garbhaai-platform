"""
Phase 9 -- the first end-to-end smoke test (runbook's own phrase).

Pushes one real (or synthetic) embryo image through the live stack and
checks, in order, the runbook's own ten-item Phase 9 checklist. Self-
contained on purpose (duplicates the Feast schema from
services/inference/feast_schema.py rather than importing across the
service boundary) -- same convention every other script in this repo
follows (see e.g. training/drift_engine/detect_drift.py's own schema
duplication).

What this script checks directly, by calling the real services and
querying the real Postgres/Feast/MLflow state:
  1. A row appears in Postgres (`images`) and the file lands in RustFS.
  2. A `quality_checks` row is written (Phase 9 gap-closure -- see
     services/quality-check/main.py).
  4. The feature vector is readable back from Feast's online store.
  5. The inference API returns a grade + confidence.
  6. A doctor review ("accept" by default, or --override GRADE) writes
     to `grade_records`.
  8. The MLflow registry has a model version holding the "production"
     alias, with its run's logged metrics.

What this script checks as a lighter, read-only platform check (these
don't depend on today's single image, so they're reported but don't
fail the run the way 1/2/4/5/6/8 do):
  9. The Argo Rollout for inference is Healthy (not stuck mid-canary).
 10. Grafana/Prometheus are reachable (manual check -- see printed URL)
     and the drift_results table has a recent row.

Item 3 (a raw/clean RustFS bucket split for preprocessing's output) and
item 7 (a nightly DVC job versioning new reviews) are deliberately out
of scope here -- see the repo's Phase 9 gap-analysis discussion for why
(preprocessing's output isn't consumed by anything downstream right
now; DVC-versioning reviews is a separate, deferred decision).

Usage:
    pip install -r requirements.txt
    python run_smoke_test.py [--image /path/to/embryo.jpg] [--day 3]
                              [--patient-id SMOKE_TEST_PATIENT]
                              [--override GRADE]

With no --image, generates a synthetic (but readable, non-trivial)
JPEG so the script can run with zero external inputs -- fine for
exercising the pipeline's plumbing, which is this phase's actual goal
per the runbook ("it does not yet need a trained model that's
clinically accurate ... it needs the pipeline to be real").
"""
import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg2
import psycopg2.extras
import requests
from PIL import Image

# --- service URLs -----------------------------------------------------------
# Defaults match services/docker-compose.yml's published host ports for
# quality-check/ingestion, and infra/k8s/inference/service.yaml for
# inference (ClusterIP -- needs `kubectl port-forward svc/garbhaai-inference
# -n garbhaai 8005:80` running in another terminal first, same port the
# local docker-compose inference container would otherwise use).
QUALITY_CHECK_URL = os.environ.get("QUALITY_CHECK_URL", "http://localhost:8002")
INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8001")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8005")

# --- Postgres (infra/local/docker-compose.yml) -------------------------------
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "garbhaai")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "garbhaai")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "garbhaai_local_dev")

# --- Feast / Redis (services/inference/feast_schema.py, duplicated) ---------
FEAST_REDIS_HOST = os.environ.get("FEAST_REDIS_HOST", "localhost")
FEAST_REDIS_PORT = os.environ.get("FEAST_REDIS_PORT", "6379")
FEAST_PROJECT = "garbhaai"
FEAST_REGISTRY_PATH = os.environ.get(
    "FEAST_REGISTRY_PATH", os.path.join(os.path.dirname(__file__), "feast_registry.db")
)

# --- MLflow (infra/local/docker-compose.yml) ---------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_REGISTRY_NAME = "garbhaai-day-classifier"
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")

GRAFANA_URL_HINT = (
    "http://localhost:3000 (after kubectl port-forward -n monitoring "
    "svc/kube-prometheus-stack-grafana 3000:80)"
)
DRIFT_FRESHNESS_WARN_HOURS = 8  # run_drift_check.sh's crontab example runs every 4h


class CheckFailed(Exception):
    """Raised by a required check (items 1/2/4/5/6/8) that didn't pass."""


def _step(n, title):
    print(f"\n[{n}] {title}")
    print("-" * (len(title) + 6))


def _ok(msg):
    print(f"  OK  {msg}")


def _warn(msg):
    print(f"  WARN  {msg}")


def _fail(msg):
    print(f"  FAIL  {msg}")


def make_synthetic_image() -> bytes:
    """A non-trivial (not flat gray, not black) synthetic JPEG, so it
    doesn't trip quality-check's blur/exposure warnings unnecessarily --
    this is about exercising the plumbing, not grading a real embryo."""
    rng = np.random.RandomState(42)
    arr = (rng.rand(224, 224, 3) * 255).astype(np.uint8)
    # a soft gradient on top of noise keeps it readable as "a photo", not
    # pure static, without needing a real sample image
    gradient = np.linspace(60, 200, 224, dtype=np.uint8)
    arr = np.clip(arr.astype(int) // 2 + gradient[None, :, None], 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def get_db_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def check_health(name, url):
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
        _ok(f"{name} is up ({url}/health -> {resp.json()})")
        return True
    except Exception as exc:
        _fail(f"{name} not reachable at {url}: {exc}")
        return False


def step_quality_check(image_bytes: bytes, filename: str) -> dict:
    _step(2, "Quality-check: run the check, confirm it's logged (item 2)")
    resp = requests.post(
        f"{QUALITY_CHECK_URL}/check-quality",
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    _ok(f"POST /check-quality -> ok={body['ok']} warnings={body['warnings']}")
    return body


def step_ingest(image_bytes: bytes, filename: str, patient_id: str, day: int) -> dict:
    _step(1, "Ingestion: Postgres row + RustFS object (item 1)")
    resp = requests.post(
        f"{INGESTION_URL}/ingest",
        data={"patient_id": patient_id, "day": day},
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    _ok(
        f"POST /ingest -> image_hash={body['image_hash']} "
        f"storage_key={body['storage_key']} already_seen={body['already_seen']}"
    )
    return body


def step_grade(image_bytes: bytes, filename: str, day: int) -> dict:
    _step(5, "Inference: grade + confidence (item 5)")
    resp = requests.post(
        f"{INFERENCE_URL}/grade",
        data={"day": day},
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    _ok(
        f"POST /grade -> source={body['source']} grade={body['grade']} "
        f"confidence={body['confidence']:.3f}"
    )
    return body


def step_review(
    grade_body: dict, image_hash: str, day: int, patient_id: str, override: str
) -> dict:
    _step(6, "Doctor review: accept/override writes to grade_records (item 6)")
    final_grade = override or grade_body["grade"]
    doctor_overridden = bool(override) and override != grade_body["grade"]
    resp = requests.post(
        f"{INFERENCE_URL}/review",
        data={
            "image_hash": image_hash,
            "day": day,
            "model_grade": grade_body["grade"],
            "model_confidence": grade_body["confidence"],
            "model_probabilities": json.dumps(grade_body["probabilities"]),
            "final_grade": final_grade,
            "doctor_overridden": doctor_overridden,
            "patient_id": patient_id,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    _ok(
        f"POST /review -> final_grade={body['final_grade']} "
        f"doctor_overridden={body['doctor_overridden']}"
    )
    return body


def step_verify_postgres(image_hash: str):
    _step("1b/2b/6b", "Postgres: confirm the images / quality_checks / grade_records rows")
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM images WHERE image_hash = %s ORDER BY ingested_at DESC LIMIT 1",
                (image_hash,),
            )
            images_row = cur.fetchone()
            if images_row:
                _ok(
                    f"images row: image_id={images_row['image_id']} "
                    f"storage_key={images_row['storage_key']}"
                )
            else:
                raise CheckFailed(f"no `images` row found for image_hash={image_hash}")

            cur.execute(
                "SELECT * FROM quality_checks WHERE image_hash = %s "
                "ORDER BY checked_at DESC LIMIT 1",
                (image_hash,),
            )
            qc_row = cur.fetchone()
            if qc_row:
                _ok(f"quality_checks row: ok={qc_row['ok']} warnings={qc_row['warnings']}")
            else:
                raise CheckFailed(f"no `quality_checks` row found for image_hash={image_hash}")

            cur.execute(
                "SELECT * FROM grade_records WHERE image_hash = %s LIMIT 1",
                (image_hash,),
            )
            review_row = cur.fetchone()
            if review_row:
                _ok(
                    f"grade_records row: final_grade={review_row['final_grade']} "
                    f"doctor_overridden={review_row['doctor_overridden']}"
                )
            else:
                raise CheckFailed(f"no `grade_records` row found for image_hash={image_hash}")
    finally:
        conn.close()


def step_verify_feast(image_hash: str):
    _step(4, "Feast: feature vector is readable back from the online store (item 4)")
    try:
        from datetime import timedelta as _td

        from feast import Entity, FeatureView, Field, FileSource
        from feast.repo_config import RepoConfig
        from feast.types import Array, Float32, Int64
        from feast.value_type import ValueType
        from feast import FeatureStore
        import pandas as pd

        embryo_image = Entity(
            name="image_id",
            value_type=ValueType.STRING,
            description="Unique identifier for a single embryo image.",
        )
        offline_path = os.path.join(os.path.dirname(__file__), "feast_offline_placeholder.parquet")
        if not os.path.exists(offline_path):
            pd.DataFrame([{
                "image_id": "unused",
                "event_timestamp": datetime.now(timezone.utc),
                "day": 0,
                "model_confidence": 0.0,
                "embedding": [0.0] * 512,
            }]).to_parquet(offline_path)
        image_features_source = FileSource(path=offline_path, timestamp_field="event_timestamp")
        image_features_view = FeatureView(
            name="image_features",
            entities=[embryo_image],
            ttl=_td(days=365),
            schema=[
                Field(name="day", dtype=Int64),
                Field(name="model_confidence", dtype=Float32),
                Field(name="embedding", dtype=Array(Float32)),
            ],
            online=True,
            source=image_features_source,
        )
        config = RepoConfig(
            project=FEAST_PROJECT,
            provider="local",
            registry=FEAST_REGISTRY_PATH,
            online_store={
                "type": "redis",
                "connection_string": f"{FEAST_REDIS_HOST}:{FEAST_REDIS_PORT}",
            },
            offline_store={"type": "file"},
            entity_key_serialization_version=3,
        )
        store = FeatureStore(config=config)
        store.apply([embryo_image, image_features_view])

        result = store.get_online_features(
            features=[
                "image_features:embedding",
                "image_features:day",
                "image_features:model_confidence",
            ],
            entity_rows=[{"image_id": image_hash}],
        ).to_dict()

        embedding = result["embedding"][0]
        if embedding is None:
            raise CheckFailed(
                f"Feast returned no embedding for image_id={image_hash} -- "
                f"either _push_to_feast() failed (check the inference pod's logs) "
                f"or FEAST_REDIS_HOST/PORT here don't point at the same Redis the "
                f"inference pod uses."
            )
        _ok(
            f"get_online_features(image_id={image_hash}) -> "
            f"embedding_dim={len(embedding)} day={result['day'][0]} "
            f"model_confidence={result['model_confidence'][0]:.3f}"
        )
    except CheckFailed:
        raise
    except Exception as exc:
        raise CheckFailed(f"could not query Feast: {exc}")


def step_verify_mlflow():
    _step(8, "MLflow: registry has a 'production' alias with logged metrics (item 8)")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, MODEL_ALIAS)
        run = client.get_run(version.run_id)
        metric_names = sorted(run.data.metrics.keys())
        if not metric_names:
            raise CheckFailed(
                f"model version {version.version} (alias '{MODEL_ALIAS}') has a run "
                f"({version.run_id}) but it has no logged metrics"
            )
        _ok(
            f"{MODEL_REGISTRY_NAME}@{MODEL_ALIAS} -> version {version.version}, "
            f"run {version.run_id}, metrics logged: {metric_names}"
        )
    except CheckFailed:
        raise
    except Exception as exc:
        raise CheckFailed(f"could not query MLflow at {MLFLOW_TRACKING_URI}: {exc}")


def step_check_rollout():
    _step(9, "Argo Rollout: inference canary is healthy, not stuck (item 9, informational)")
    try:
        result = subprocess.run(
            [
                "kubectl", "get", "rollout", "garbhaai-inference", "-n", "garbhaai",
                "-o", "jsonpath={.status.phase}",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            _warn(
                "could not read Rollout status (is the kind cluster up? "
                f"kubectl said: {result.stderr.strip()})"
            )
            return
        phase = result.stdout.strip()
        if phase == "Healthy":
            _ok(f"garbhaai-inference Rollout phase = {phase}")
        else:
            _warn(
                f"garbhaai-inference Rollout phase = {phase!r} "
                "(expected 'Healthy' between promotions)"
            )
    except FileNotFoundError:
        _warn("kubectl not found on PATH -- skipping (informational check only)")
    except Exception as exc:
        _warn(f"could not check Rollout status: {exc}")


def step_check_monitoring():
    _step(10, "Monitoring: Grafana reachability + drift job freshness (item 10, informational)")
    print(f"  Grafana (check manually for request metrics): {GRAFANA_URL_HINT}")
    try:
        conn = get_db_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT checked_at, drift_flagged, skipped FROM drift_results "
                    "ORDER BY checked_at DESC LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            _warn(
                "drift_results has no rows yet -- run "
                "training/drift_engine/detect_drift.py at least once"
            )
            return
        age = datetime.now(timezone.utc) - row["checked_at"]
        if age > timedelta(hours=DRIFT_FRESHNESS_WARN_HOURS):
            _warn(
                f"most recent drift_results row is {age} old "
                f"(checked_at={row['checked_at']}) -- older than "
                f"{DRIFT_FRESHNESS_WARN_HOURS}h; confirm the cron job is actually firing"
            )
        else:
            _ok(f"most recent drift_results row: checked_at={row['checked_at']} ({age} ago)")
    except Exception as exc:
        _warn(f"could not query drift_results: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", help="path to a real embryo image; omit to use a synthetic one")
    parser.add_argument("--day", type=int, default=3, choices=[3, 4])
    parser.add_argument("--patient-id", default="SMOKE_TEST_PATIENT")
    parser.add_argument(
        "--override",
        default=None,
        help="doctor's chosen grade (A/B/C); omit to accept the model's grade",
    )
    args = parser.parse_args()

    if args.image:
        with open(args.image, "rb") as f:
            image_bytes = f.read()
        filename = os.path.basename(args.image)
    else:
        print("No --image given -- generating a synthetic test image.")
        image_bytes = make_synthetic_image()
        filename = "smoke_test_synthetic.jpg"

    print("=" * 70)
    print("GarbhaAI Phase 9 smoke test")
    print("=" * 70)

    _step(0, "Health checks")
    all_up = True
    for name, url in [
        ("quality-check", QUALITY_CHECK_URL),
        ("ingestion", INGESTION_URL),
        ("inference", INFERENCE_URL),
    ]:
        all_up = check_health(name, url) and all_up
    if not all_up:
        print(
            "\nOne or more services aren't reachable. If inference is the one that "
            "failed, you likely need:\n"
            "  kubectl port-forward svc/garbhaai-inference -n garbhaai 8005:80\n"
            "running in another terminal first (it's a ClusterIP Service)."
        )
        sys.exit(1)

    try:
        step_quality_check(image_bytes, filename)
        ingest_body = step_ingest(image_bytes, filename, args.patient_id, args.day)
        image_hash = ingest_body["image_hash"]

        grade_body = step_grade(image_bytes, filename, args.day)
        step_review(grade_body, image_hash, args.day, args.patient_id, args.override)

        step_verify_postgres(image_hash)

        if grade_body["source"] == "model":
            step_verify_feast(image_hash)
        else:
            _step(4, "Feast: feature vector readback (item 4)")
            _warn(
                f"source={grade_body['source']!r}, not 'model' -- this image hash was already "
                f"reviewed before (cached_review short-circuit in main.py's /grade), so this run "
                f"never called feature-extraction or pushed to Feast. Re-run with a different "
                f"--image (or no --image, for a fresh synthetic one) to exercise this check."
            )

        step_verify_mlflow()
        step_check_rollout()
        step_check_monitoring()

    except CheckFailed as exc:
        print(f"\n{'=' * 70}\nSMOKE TEST FAILED: {exc}\n{'=' * 70}")
        sys.exit(1)
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        print(f"\n{'=' * 70}\nSMOKE TEST FAILED: HTTP error -- {exc}\n{body}\n{'=' * 70}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("Required checks (items 1, 2, 4*, 5, 6, 8) all passed.")
    print("Items 9 and 10 are informational -- review any WARN lines above.")
    print("Items 3 and 7 are deliberately out of scope (see module docstring).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
