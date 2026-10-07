# Phase 4/5 -- application services

Each service is self-contained on purpose (no shared library between them):
that's a deliberate microservices choice, not an oversight -- see the
module docstring at the top of each `main.py`. Where one service needs
another's work (inference needs feature-extraction's output, for instance)
it calls that service's HTTP API rather than importing its code -- that's
the correct way to reuse work across independently-deployable services.

## Services and API contracts

| Service | Port (compose) | Endpoint | Ported from |
|---|---|---|---|
| `quality-check` | 8002 | `POST /check-quality` (multipart `file`) -> `{ok, warnings}` | `grading_core.check_image_quality` |
| `preprocessing` | 8003 | `POST /preprocess` (multipart `file`) -> `{shape, dtype, tensor_b64}` | `grading_core.preprocess_for_model` |
| `feature-extraction` | 8004 | `POST /extract-features` (multipart `file`) -> `{feature_dim, features}` | new -- backbone split out of the old fused classifier |
| `ingestion` | 8001 | `POST /ingest` (form `patient_id`, `day` + multipart `file`) -> `{image_id, image_hash, storage_key, already_seen}` | `records_store` (local JSON/disk -> Postgres + RustFS) |
| `inference` | 8005 | `POST /grade` (form `day` + multipart `file`) -> `{source, image_hash, grade, confidence, probabilities}`; `POST /review` (form fields) -> `{image_hash, final_grade, doctor_overridden}` | `grading_core`'s classifier head + `records_store`'s review/cache logic |
| `ui` (`ui-next/`) | 8511 | Next.js app; its server routes forward to the above over HTTP | `app/page.tsx` (thin client, no business logic) |

All services also expose `GET /health`.

### `inference` in detail

- `POST /grade`: computes the image's hash; if a doctor has already reviewed
  this exact image (`grade_records` table), returns that confirmed grade
  instantly (`source: "cached_review"`) without touching the model at all --
  this is the "never contradict a decision we've already made on an image
  we've seen" behavior from the original app, now backed by Postgres instead
  of a local JSON file. Otherwise it calls `feature-extraction` for the
  512-dim feature vector, concatenates the day one-hot, runs the bundled
  classifier head (`model/unified_day3_day4_classifier.pt` -- just the
  `Linear(514, 3)` head; the ResNet18 backbone lives only in
  `feature-extraction`), and returns `source: "model"`.
- `POST /review`: the doctor's accept/override decision. Upserts a row into
  `grade_records` keyed by image hash -- this *is* the labeled data store.
  An "accept" sends `final_grade == model_grade, doctor_overridden=false`;
  an override sends the doctor's chosen grade with `doctor_overridden=true`.
- **Day 5 is out of scope**: the checkpoint was only ever trained on the
  pooled Day3+Day4 dataset (see Phase 3's data-quality findings for why),
  so `/grade` returns 422 for `day=5` rather than a meaningless prediction.
- The model-loading approach here (a checkpoint file bundled into the
  service's Docker image) is a deliberate placeholder -- Phase 6 (MLflow)
  is expected to replace it with a real model registry pull.

## Running locally

```bash
cd infra/local && ./bootstrap.sh        # Postgres + RustFS + Redis (Phase 2), if not already up
cd ../../services
docker compose up --build
open http://localhost:8511
```

Or hit any service directly, e.g.:

```bash
curl -F "file=@/path/to/embryo.jpg" http://localhost:8002/check-quality
curl -F "day=3" -F "file=@/path/to/embryo.jpg" http://localhost:8005/grade
```

## Tests

```bash
cd services/<name>
pip install -r requirements.txt pytest
pytest tests
```

`ingestion`, `feature-extraction`, and `inference`'s tests stub out
Postgres/RustFS, the network weight download, and Postgres/the
feature-extraction HTTP call respectively (see each `tests/test_main.py`
docstring), so they run without any external service or network access --
same as CI.
