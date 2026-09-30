# GarbhaAI feature repo (Feast)

Local dev setup: Feast's offline store is a parquet file, online store is
the Redis container from `infra/local/docker-compose.yml`. This proves
the Feast <-> Redis wiring end to end with placeholder data before the
real feature-extraction service (Phase 4) exists.

## Setup

```
pip install "feast[redis]"
cd training/feature_repo

python generate_sample_data.py      # writes data/image_features.parquet
feast apply                         # registers entity + feature view, connects to Redis
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)   # pushes features into Redis
python test_retrieval.py            # confirms a feature comes back out of Redis
```

`infra/local`'s postgres/redis containers must be running first
(`../../infra/local/bootstrap.sh`).

## What happens later

Phase 4's feature-extraction service will write real per-image features
(sharpness/blur score, embedding stats, etc.) instead of
`generate_sample_data.py`'s placeholder rows -- `features.py`'s
`FileSource` path just needs to point at that service's output, or get
swapped for a `PushSource` if the service pushes features directly.
Nothing else in this folder needs to change for that swap.
