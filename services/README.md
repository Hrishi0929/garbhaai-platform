# Phase 4 -- application services

Each service is self-contained on purpose (no shared library between them):
that's a deliberate microservices choice, not an oversight -- see the
module docstring at the top of each `main.py`. Any duplicated logic
(image normalization, for instance) is duplicated because these are meant
to be independently deployable, independently owned services, not because
nobody noticed the overlap.

## Services and API contracts

| Service | Port (compose) | Endpoint | Ported from |
|---|---|---|---|
| `quality-check` | 8002 | `POST /check-quality` (multipart `file`) -> `{ok, warnings}` | `grading_core.check_image_quality` |
| `preprocessing` | 8003 | `POST /preprocess` (multipart `file`) -> `{shape, dtype, tensor_b64}` | `grading_core.preprocess_for_model` |
| `feature-extraction` | 8004 | `POST /extract-features` (multipart `file`) -> `{feature_dim, features}` | new -- backbone split out of the old fused classifier |
| `ingestion` | 8001 | `POST /ingest` (form `patient_id`, `day` + multipart `file`) -> `{image_id, image_hash, storage_key, already_seen}` | `records_store` (local JSON/disk -> Postgres + RustFS) |
| `ui` | 8501 | Streamlit app, calls the above over HTTP | `app.py` (thin client now, no business logic) |

All four APIs also expose `GET /health`.

## What's deliberately NOT here yet (Phase 5)

- The classifier head (day-conditioned grade + confidence) -- `inference` service.
- The image-hash re-grading cache (same image bytes -> same answer, regardless
  of patient ID) -- also `inference`, since it's the service that decides
  whether to trust a cached grade.
- Doctor accept/override review flow.

The `ui` service here stops at "features extracted" and says so on screen.

## Running locally

```bash
cd infra/local && ./bootstrap.sh        # Postgres + RustFS + Redis (Phase 2), if not already up
cd ../../services
docker compose up --build
open http://localhost:8501
```

Or hit any service directly, e.g.:

```bash
curl -F "file=@/path/to/embryo.jpg" http://localhost:8002/check-quality
```

## Tests

```bash
cd services/<name>
pip install -r requirements.txt pytest
pytest tests
```

`ingestion` and `feature-extraction`'s tests stub out Postgres/RustFS and
network weight downloads respectively (see each `tests/test_main.py` docstring)
so they run without any external service or network access -- same as CI.
