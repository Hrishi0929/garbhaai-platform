#!/usr/bin/env bash
# Phase 2 exit-gate helper: brings up postgres/minio/redis.
# MinIO auto-creates the garbhaai-images bucket on first boot via
# MINIO_DEFAULT_BUCKETS in docker-compose.yml.
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

echo "==> Done. Services:"
echo "  Postgres  -> localhost:5432  (user: garbhaai / db: garbhaai)"
echo "  MinIO API -> localhost:9000  (console: http://localhost:9001, bucket: garbhaai-images)"
echo "  Redis     -> localhost:6379"
