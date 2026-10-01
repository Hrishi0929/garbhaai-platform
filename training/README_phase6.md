# Phase 6 -- MLflow server + training/eval engines

## What's new

- **MLflow tracking server** (`infra/local/mlflow`): self-hosted, built from
  a small custom image (official `mlflow` package has no Postgres driver or
  S3 client built in). Backend store = the same Postgres as everything else
  (Phase 2); artifact store = RustFS under the `mlflow-artifacts/` prefix of
  the existing `garbhaai-images` bucket. No new infrastructure, reuses what
  Phase 2 already stood up. Runs at `http://localhost:5500`.
- **`training/training_engine/train.py`**: ported from the pre-platform
  `garbhaai-embryo-grading/src/train_unified.py`. Same model (frozen
  ResNet18 + day one-hot -> `Linear(514, 3)`), same leave-one-out CV over
  the pooled Day3+Day4 dataset, same final-model-trained-on-everything
  checkpoint format -- but features now come from the live Phase 4
  `feature-extraction` service instead of a second copy of the backbone
  living in the training script, and every run is logged to the new MLflow
  server (params, per-fold and per-class metrics, the metrics JSON, and the
  checkpoint itself) and registered in the MLflow Model Registry as
  `garbhaai-day-classifier`.
- **`training/eval_engine/evaluate.py`**: a pass/fail gate over a training
  run, meant to run right after `train.py` and before Phase 7 deploys
  anything. Checks checkpoint structure (does it actually match what
  `services/inference` expects?), metric completeness (nothing NaN or
  missing), and reports accuracy with a WARNING (not a hard fail) below
  chance level -- consistent with this project's own documented caveat that
  LOOCV over ~30-45 images is a noisy, small-sample estimate, not a precise
  number to gate hard on.

## Running it

```bash
cd infra/local && ./bootstrap.sh              # now also builds + starts mlflow
cd ../../services && docker compose up -d     # needs feature-extraction reachable
cd ../training/training_engine
pip install -r requirements.txt
python train.py
```

Then evaluate the run it just produced:

```bash
cd ../eval_engine
pip install -r requirements.txt
python evaluate.py          # evaluates the latest run
# or: python evaluate.py --run-id <run_id>
```

View everything at `http://localhost:5500`.

## What Phase 6 sets up for Phase 7

`services/inference` currently loads its model from a checkpoint file
bundled into its Docker image (`model/unified_day3_day4_classifier.pt`) --
flagged as a deliberate placeholder back in Phase 5. Now that there's a real
model registry (`garbhaai-day-classifier` in MLflow), Phase 7's deployment
gate is expected to: run `train.py`, run `evaluate.py` as a gate, and on
pass, have `inference` pull the registered model version at startup instead
of using the bundled file.

## Tests

`eval_engine`'s structural/completeness checks are pure functions with no
MLflow or network dependency, so they're unit-tested and run in CI
(`training-eval-engine` job) -- including one test that validates the real
checkpoint bundled in `services/inference` against this gate, so the test
suite itself proves the gate wouldn't reject what's actually deployed.
`train.py` has no automated tests (it requires a live MLflow server and
feature-extraction service to do anything), so CI only lints it.
