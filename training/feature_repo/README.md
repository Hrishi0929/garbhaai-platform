# GarbhaAI feature repo (Feast)

Local dev setup: Feast's offline store is a parquet file, online store is
the Redis container from `infra/local/docker-compose.yml`.

## Setup (smoke-testing the Feast <-> Redis wiring in isolation)

```
pip install "feast[redis]==0.47.0"
cd training/feature_repo

python generate_sample_data.py      # writes data/image_features.parquet
feast apply                         # registers entity + feature view, connects to Redis
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)   # pushes placeholder rows into Redis
python test_retrieval.py            # confirms a feature (incl. the 512-dim embedding) comes back out of Redis
```

`infra/local`'s postgres/redis containers must be running first
(`../../infra/local/bootstrap.sh`).

## What actually populates this in production (Phase 8)

The placeholder flow above is only for testing this folder in isolation.
The real data path is: `services/inference/main.py` extracts a 512-dim
ResNet18 embedding for every image it grades (via the feature-extraction
service), then calls Feast's `write_to_online_store()` directly --
live, per-request -- to push that embedding (plus `day` and
`model_confidence`) straight into the same Redis online store, keyed by
the image's hash as `image_id`. It never goes through
`generate_sample_data.py`, `feast apply`, or `feast materialize`; those
remain useful only for testing this folder on its own.

`services/inference/feast_schema.py` keeps its own copy of this file's
`Entity`/`FeatureView` definitions (each service in this repo stays
self-contained rather than importing across a service boundary -- see
`services/inference/main.py`'s docstring) and builds its `FeatureStore`
purely in Python (a `RepoConfig` object, no `feature_store.yaml` file),
pointed at the real Redis host via the `FEAST_REDIS_HOST`/
`FEAST_REDIS_PORT` env vars so it works whether the service is running
on the host, in Docker Compose, or in the kind cluster.

`training/drift_engine/detect_drift.py` is the consumer on the other
end: it reads those live-pushed embeddings back out of the online store
and compares their distribution against a training-set baseline.
