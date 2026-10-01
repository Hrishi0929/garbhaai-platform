#!/usr/bin/env python3
"""
Phase 6 eval engine: a pass/fail gate over a training run, meant to run
right after training_engine/train.py and before Phase 7 ever considers
deploying that run's model.

Checks three things, in order of how badly a failure should be treated:

1. STRUCTURAL -- does the logged checkpoint artifact actually match the
   shape the inference service expects (Linear(514, 3), class_names ==
   ["A","B","C"], day_onehot has day3/day4 keys)? A malformed checkpoint
   here would make it into services/inference and fail at request time
   instead of at this gate -- this is the check that actually prevents
   that. HARD FAIL.
2. COMPLETENESS -- are the metrics this script needs (mean_accuracy,
   per-class precision/recall/f1) actually present and finite (no NaN)?
   A run that crashed partway through logging would still "exist" in
   MLflow but be unusable as a gate input. HARD FAIL.
3. ACCURACY -- is mean_accuracy reasonable? Given the project's own
   documented caveat (LOOCV over ~30-45 images is a noisy, small-sample
   estimate -- see train.py and the original train_unified.py), this is
   a WARNING below chance level (1/3 for 3 classes), not a hard block.
   Silently blocking on a noisy metric would be dishonest about what
   this number actually means at this sample size.

Usage:
  python evaluate.py                   # evaluates the latest run in the experiment
  python evaluate.py --run-id <run_id> # evaluates a specific run

Exit code 0 = pass (warnings allowed), 1 = hard fail.
"""
import argparse
import math
import os
import sys
import tempfile

import mlflow
import torch
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MLFLOW_EXPERIMENT = "garbhaai_embryo_grading"

EXPECTED_CLASS_NAMES = ["A", "B", "C"]
EXPECTED_DAY_KEYS = {"day3", "day4"}
EXPECTED_HEAD_SHAPE = (3, 514)  # 3 classes x (512 features + 2-dim day one-hot)
CHANCE_LEVEL = 1 / 3
REQUIRED_METRICS = [
    "mean_accuracy", "final_train_loss",
    "A_precision", "A_recall", "A_f1",
    "B_precision", "B_recall", "B_f1",
    "C_precision", "C_recall", "C_f1",
]


def get_latest_run_id(client: MlflowClient) -> str:
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise SystemExit(
            f"No MLflow experiment named '{MLFLOW_EXPERIMENT}' found at {MLFLOW_TRACKING_URI}"
        )
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    if not runs:
        raise SystemExit(f"No runs found in experiment '{MLFLOW_EXPERIMENT}'")
    return runs[0].info.run_id


def check_completeness(metrics: dict) -> list:
    failures = []
    for name in REQUIRED_METRICS:
        if name not in metrics:
            failures.append(f"missing required metric: {name}")
            continue
        if math.isnan(metrics[name]):
            failures.append(f"metric {name} is NaN")
    return failures


def validate_checkpoint(ckpt: dict) -> list:
    """Pure (no MLflow/network dependency) so it's directly unit-testable,
    including against the real checkpoint bundled in services/inference."""
    failures = []
    if ckpt.get("class_names") != EXPECTED_CLASS_NAMES:
        got = ckpt.get("class_names")
        failures.append(f"class_names {got} != expected {EXPECTED_CLASS_NAMES}")
    day_keys = set(ckpt.get("day_onehot", {}).keys())
    if day_keys != EXPECTED_DAY_KEYS:
        failures.append(f"day_onehot keys {day_keys} != expected {EXPECTED_DAY_KEYS}")
    state_dict = ckpt.get("classifier_state_dict", {})
    weight_shape = tuple(state_dict.get("weight", torch.empty(0)).shape)
    if weight_shape != EXPECTED_HEAD_SHAPE:
        msg = f"classifier head weight shape {weight_shape} != expected {EXPECTED_HEAD_SHAPE}"
        failures.append(msg)
    return failures


def check_structure(client: MlflowClient, run_id: str) -> list:
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    checkpoint_name = next((a for a in artifacts if a.endswith(".pt")), None)
    if checkpoint_name is None:
        return [f"no .pt checkpoint artifact found on run {run_id} (artifacts: {artifacts})"]

    with tempfile.TemporaryDirectory() as tmp:
        local_path = client.download_artifacts(run_id, checkpoint_name, tmp)
        ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    return validate_checkpoint(ckpt)


def evaluate(run_id: str) -> bool:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    print(f"Evaluating run {run_id} ({run.info.run_name})")

    structure_failures = check_structure(client, run_id)
    completeness_failures = check_completeness(metrics)
    hard_failures = structure_failures + completeness_failures

    for f in hard_failures:
        print(f"FAIL: {f}")

    if hard_failures:
        print(f"\n{len(hard_failures)} hard failure(s) -- this run should NOT be deployed.")
        return False

    mean_acc = metrics["mean_accuracy"]
    print(f"mean_accuracy = {mean_acc:.4f} (chance level for 3 classes = {CHANCE_LEVEL:.4f})")
    if mean_acc < CHANCE_LEVEL:
        print(
            "WARNING: below chance level. At this dataset size (LOOCV over "
            "~30-45 images, per the project's own documented caveat) this is "
            "plausible noise, not necessarily a broken model -- not failing "
            "the gate on this alone, but a human should look before rollout."
        )

    print("\nPASS -- structure and metrics are sound.")
    print("See warnings above, if any, before deploying.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=None, help="MLflow run id to evaluate (default: latest run)"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run_id = args.run_id or get_latest_run_id(client)

    ok = evaluate(run_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
