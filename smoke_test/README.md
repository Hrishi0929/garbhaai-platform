# Phase 9 -- end-to-end smoke test

Runs the runbook's own Phase 9 checklist against the real, live stack: one
image through quality-check, ingestion, inference, a doctor review, and
direct reads of Postgres/Feast/MLflow/kubectl/Grafana to confirm each step
actually left the evidence the checklist expects.

## Prerequisites

Everything needs to actually be running first:

```bash
# Phase 2 data plane (Postgres, RustFS, Redis, MLflow)
cd infra/local && ./bootstrap.sh

# Phase 4/5 application services (quality-check, ingestion, preprocessing,
# feature-extraction, ui) -- NOT inference, which runs on kind (Phase 7)
cd ../../services && docker compose up --build

# Phase 7: inference runs on the kind cluster, not docker-compose. Forward
# its ClusterIP Service to localhost so this script can reach it the same
# way it reaches the other services:
kubectl port-forward svc/garbhaai-inference -n garbhaai 8005:80
```

A model version needs the `production` alias set (Phase 6 training +
`training/promote_engine/promote.py`) before inference can grade anything
-- if `/grade` 500s with "no model loaded", that's why.

## Running it

```bash
cd smoke_test
pip install -r requirements.txt

# with a real image:
python run_smoke_test.py --image /path/to/embryo.jpg --day 3

# or with no image at all -- generates a synthetic one, useful for
# exercising the pipeline's plumbing without a real sample on hand:
python run_smoke_test.py
```

Use `--override B` to simulate a doctor overriding the model's grade
(exercises `doctor_overridden=True` in the `grade_records` row); omit it
to simulate the doctor simply accepting the model's grade.

Env vars (all optional, shown with their defaults) let you point the
script at a different host/port for any piece, e.g. if you've forwarded
inference to a different local port:

| var | default |
|---|---|
| `QUALITY_CHECK_URL` | `http://localhost:8002` |
| `INGESTION_URL` | `http://localhost:8001` |
| `INFERENCE_URL` | `http://localhost:8005` |
| `POSTGRES_HOST` / `_PORT` / `_DB` / `_USER` / `_PASSWORD` | `localhost` / `5432` / `garbhaai` / `garbhaai` / `garbhaai_local_dev` |
| `FEAST_REDIS_HOST` / `_PORT` | `localhost` / `6379` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5500` |
| `MODEL_ALIAS` | `production` |

## What it checks, against the runbook's own Phase 9 list

| # | Runbook item | How this script checks it |
|---|---|---|
| 1 | Postgres row + RustFS object on ingest | `POST /ingest`, then `SELECT` from `images` |
| 2 | `quality_checks` row for the trace_id | `POST /check-quality`, then `SELECT` from `quality_checks` (Phase 9 gap-closure) |
| 3 | Processed copy in a "clean" bucket | **Out of scope** -- preprocessing is stateless and nothing downstream reads a clean bucket; deferred per the Phase 9 gap-analysis decision |
| 4 | Feature vector readable from Feast | builds a `FeatureStore` from the same schema as `services/inference/feast_schema.py`, calls `get_online_features(image_id=<image_hash>)` |
| 5 | Inference returns grade + confidence | `POST /grade` |
| 6 | Doctor review writes `app_db.labels` | `POST /review`, then `SELECT` from `grade_records` (this repo's name for that table) |
| 7 | Nightly DVC job versions the new label | **Out of scope** -- no scheduled DVC job exists yet; deferred per the Phase 9 gap-analysis decision |
| 8 | Training run + registry version in MLflow | `MlflowClient.get_model_version_by_alias(..., "production")`, checks the run has logged metrics |
| 9 | Promotion updates the live traffic split | informational: `kubectl get rollout garbhaai-inference -n garbhaai` phase check |
| 10 | Grafana shows metrics; drift job runs on schedule | informational: prints the Grafana URL to check by hand, and checks `drift_results` for a recent row |

Items 1, 2, 4, 5, 6, and 8 are hard requirements -- the script exits
non-zero and prints `SMOKE TEST FAILED` if any of them don't check out.
Items 9 and 10 only print `WARN` lines, since they don't depend on this
one image and failing the whole run over a stale Grafana dashboard isn't
useful.

Items 3 and 7 are the two gaps the project explicitly decided to defer
after the Phase 9 gap-analysis discussion, to revisit once this smoke
test itself is running clean -- they're listed above for completeness,
not because this script checks them.

## A note on item 4 and repeat runs

Inference's `/grade` endpoint short-circuits to a cached review
(`source != "model"`) if it's seen this exact image hash before, which
skips the feature-extraction + Feast-push path entirely. Since the
synthetic image is deterministic, a second run with no `--image` will
print a `WARN` instead of exercising the Feast check. Pass a different
real `--image` (or add `np.random.RandomState(42)`'s seed as a CLI flag,
if this comes up often enough to be worth doing) to get a fresh hash on
every run.
