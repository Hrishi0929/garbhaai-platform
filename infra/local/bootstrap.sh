#!/usr/bin/env bash
# Phase 2 exit-gate helper: brings up postgres/minio/redis and creates
# the MinIO bucket the pipeline will write images/artifacts to.
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Starting postgres, minio, redis..."
docker compose up -d

echo "==> Waiting for services to report healthy..."
for i in $(seq 1 30); do
  unhealthy=$(docker compose ps --format '{{.Service}} {{.Health}}' | grep -v healthy || true)
  if [ -z "$unhealthy" ]; then
    echo "All services healthy."
    break
  fi
  sleep 2
done

echo "==> Creating MinIO bucket (garbhaai-images) if it doesn't exist..."
docker run --rm --network host \
  -e MC_HOST_local="http://garbhaai:garbhaai_local_dev@localhost:9000" \
  quay.io/minio/mc mb -p local/garbhaai-images

echo "==> Done. Services:"
echo "  Postgres  -> localhost:5432  (user: garbhaai / db: garbhaai)"
echo "  MinIO API -> localhost:9000  (console: http://localhost:9001)"
echo "  Redis     -> localhost:6379"
