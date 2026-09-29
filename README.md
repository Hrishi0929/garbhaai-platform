# GarbhaAI Platform

Zero-cost, self-hosted build-out of the GarbhaAI MLOps pipeline, following
`GarbhaAI Pipeline — Bring-Up & Installation Runbook` (28 Sept 2026). Every
paid service in the original architecture has a free/self-hosted stand-in
with the same role, so the pipeline's shape doesn't change -- only where
each piece runs. Swap list:

| Role in the diagram      | Paid (later, with billing)   | Free (this repo)                                   |
|---------------------------|-------------------------------|------------------------------------------------------|
| Object storage             | Cloud Storage                 | MinIO (Docker Compose)                               |
| Operational database       | Cloud SQL (Postgres)          | Postgres (Docker Compose)                            |
| Feature Store               | Vertex AI Feature Store       | Feast (open-source) + Redis                          |
| Container registry          | Artifact Registry              | GitHub Container Registry (ghcr.io)                   |
| CD target                   | GKE + Cloud Run                | kind (Kubernetes-in-Docker)                            |
| Compute for services         | Cloud Run                      | Docker Compose / containers on kind                    |
| Scheduler                    | Cloud Scheduler                | cron / APScheduler / scheduled GitHub Actions          |
| Pub/Sub                       | Pub/Sub                        | Redis pub/sub                                          |
| Training compute              | Vertex AI Custom Training      | Local machine, or free Colab GPU                       |
| Monitoring                     | Cloud Monitoring               | Prometheus + Grafana (Docker/Helm)                      |

GitHub, DVC, GitHub Actions, Great Expectations, OpenLineage, Marquez,
MLflow, and Argo CD are unchanged either way -- free/open-source whatever
they run on.

## Layout

- `services/` -- one folder per containerized service (UI, ingestion,
  quality-check, preprocessing, feature-extraction, inference, ...), each
  with its own Dockerfile and tests, matching the diagram's numbered
  boxes.
- `infra/` -- Kubernetes manifests / Helm values / Docker Compose files
  for everything the services run on or talk to (Postgres, MinIO, Redis,
  Feast, Marquez, Argo CD, Prometheus/Grafana).
- `training/` -- training & evaluation code (MLflow-tracked).
- `.github/workflows/` -- CI (build/test/push per service) and any
  scheduled jobs (drift detection).

## Phases

Tracked in the project's task list, same order as the runbook: 0
Foundations -> 1 Version control & CI -> 2 Core data infra -> 3 Data
quality & lineage -> 4 Application services -> 5 Inference path -> 6
Training & experimentation -> 7 Deployment/CD -> 8 Monitoring & drift ->
9 End-to-end smoke test. Each phase has an explicit exit gate -- see the
runbook -- and nothing moves to the next phase until its gate is met.

## Prerequisites (Phase 0 -- run these yourself; this repo's automation
can't install them for you)

```
# Docker Desktop (Mac): https://docker.com -- the only hard requirement,
# everything else runs inside it
brew install --cask docker

# kind -- a real local Kubernetes cluster inside Docker, free
brew install kind

# kubectl + helm
brew install kubectl helm

# Git and Python 3.11 -- you already have both
```

Then create the local cluster:

```
kind create cluster --name garbhaai
kubectl cluster-info --context kind-garbhaai
```
