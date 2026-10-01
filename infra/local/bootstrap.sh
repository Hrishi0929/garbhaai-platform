#!/usr/bin/env bash
# Phase 2/6 exit-gate helper: brings up postgres/rustfs/redis/mlflow and
# creates the garbhaai-images bucket in RustFS (our S3-compatible object
# store). mlflow is built from infra/local/mlflow/Dockerfile on first run,
# which takes longer than the other services -- that's expected.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Starting postgres, rustfs, redis, mlflow..."
docker compose up -d --build

echo "==> Waiting for services to report healthy..."
for i in $(seq 1 30); do
  unhealthy=$(docker compose ps --format '{{.Service}} {{.Health}}' | grep -v healthy || true)
  if [ -z "$unhealthy" ]; then
    echo "All services healthy."
    break
  fi
  sleep 2
done

echo "==> Creating garbhaai-images bucket in RustFS (if it doesn't exist)..."
docker run --rm --network host \
  -e AWS_ACCESS_KEY_ID=garbhaai \
  -e AWS_SECRET_ACCESS_KEY=garbhaai_local_dev \
  -e AWS_DEFAULT_REGION=us-east-1 \
  amazon/aws-cli --endpoint-url http://localhost:9000 \
  s3 mb s3://garbhaai-images || echo "  (bucket likely already exists, continuing)"

echo "==> Done. Services:"
echo "  Postgres    -> localhost:5432  (user: garbhaai / db: garbhaai)"
echo "  RustFS API  -> localhost:9000  (console: http://localhost:9001, bucket: garbhaai-images)"
echo "  Redis       -> localhost:6379"
echo "  MLflow      -> http://localhost:5500"
