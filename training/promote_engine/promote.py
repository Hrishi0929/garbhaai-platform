#!/usr/bin/env python3
"""
Phase 7 validation gate / promotion script.

Moves the "production" alias of the garbhaai-day-classifier registered
model (same registry MLflow backs, same model train.py registers a new
version of on every run) from its current holder to a candidate version --
but only if the candidate clears the bar. Per the runbook:

  "promote a model from 'staging' to 'production' only after it clears a
  minimum bar on the weakest-performing class, not the average -- a model
  that's excellent at Grade A/C but poor at the harder Grade B distinction
  isn't fit for use. Every promotion is logged with who approved it."

So the gate, in order:
  1. STRUCTURAL + COMPLETENESS checks -- identical to eval_engine/
     evaluate.py's hard checks (kept as a standalone copy here rather than
     imported, since eval_engine and promote_engine are independently
     pip-installed packages with their own requirements.txt/venv, same
     pattern as the rest of this repo -- see that file for the full
     rationale of each check). HARD FAIL either way.
  2. COMPARISON -- candidate's worst-class F1 must beat the current
     production version's worst-class F1 (ties broken by mean_accuracy).
     If nothing currently holds the "production" alias, this is the first
     promotion and the candidate wins automatically.

On promotion: moves the alias (mlflow.set_registered_model_alias --
MLflow's maintained API; transition_model_version_stage is deprecated as
of 2.9.0, confirmed against the actual installed mlflow-skinny 3.16.1
source), tags the version with who approved it and when, and bumps the
model-version annotation in the inference Rollout manifest so Argo CD
picks up the change and the canary rollout (Argo Rollouts) actually ships
the new model to pods -- promoting the registry alone doesn't move any
traffic; see infra/k8s/inference/rollout.yaml's own comments for how a pod
picks up the newly-promoted version at startup.

promote.py does NOT git add/commit/push the manifest change itself --
every other commit in this project has gone through the user's own
terminal (see git history), and this is no different: it prints the exact
commands to run after it edits the file.

Usage:
  python promote.py --approved-by "Hrishi Wadki"
      # considers the highest-numbered registered version a candidate

  python promote.py --approved-by "Hrishi Wadki" --version 2
  python promote.py --approved-by "Hrishi Wadki" --run-id <mlflow_run_id>

  python promote.py --approved-by "Hrishi Wadki" --dry-run
      # runs the gate and the comparison, prints the decision, changes
      # nothing in the registry or on disk

Exit code 0 = promoted (or --dry-run would have promoted), 1 = gate
failed or candidate did not beat production.
"""
import argparse
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_REGISTRY_NAME = "garbhaai-day-classifier"
PRODUCTION_ALIAS = "production"

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "infra" / "k8s" / "inference" / "rollout.yaml"
)

# Same local-dev-only RustFS credentials as train.py/evaluate.py -- see
# train.py's comment for the full explanation. setdefault so a real env
# var (e.g. pointed at a non-local MLflow later) still wins.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "garbhaai")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "garbhaai_local_dev")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

CLASS_NAMES = ["A", "B", "C"]
EXPECTED_CLASS_NAMES = ["A", "B", "C"]
EXPECTED_DAY_KEYS = {"day3", "day4"}
EXPECTED_HEAD_SHAPE = (3, 514)
REQUIRED_METRICS = [
    "mean_accuracy", "final_train_loss",
    "A_precision", "A_recall", "A_f1",
    "B_precision", "B_recall", "B_f1",
    "C_precision", "C_recall", "C_f1",
]


# --- gate checks (standalone copy of eval_engine/evaluate.py's hard checks) ---

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


def worst_class_f1(metrics: dict) -> float:
    return min(metrics[f"{c}_f1"] for c in CLASS_NAMES)


# --- promotion logic ---

def get_version_for_run(client: MlflowClient, run_id: str, name: str):
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.run_id == run_id:
            return mv
    return None


def get_highest_version(client: MlflowClient, name: str):
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return max(versions, key=lambda mv: int(mv.version))


def get_current_production(client: MlflowClient, name: str):
    try:
        return client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def bump_manifest(path: Path, version: str, run_id: str) -> bool:
    """Rewrites the model-version annotation on the Rollout's pod template
    in place (plain regex substitution, not a YAML round-trip, so the
    manifest's comments and formatting survive -- this file is meant to be
    readable in git history/diffs). Changing a pod template annotation is
    what makes Kubernetes (and therefore Argo Rollouts) see the pod spec as
    changed and start a new canary rollout; see rollout.yaml's own comment
    on this annotation for why bumping it is what actually triggers
    anything, versus just moving the MLflow alias (which a running pod
    would never notice on its own)."""
    if not path.exists():
        print(
            f"\nNOTE: {path} does not exist yet -- skipping the manifest bump. "
            f"This is expected if infra/k8s/inference/ (Task #22) hasn't been "
            f"created yet. The registry alias has still been moved."
        )
        return False

    text = path.read_text()
    pattern = re.compile(r'(garbhaai\.io/model-version:\s*)"[^"]*"')
    new_text, n = pattern.subn(rf'\g<1>"{version}"', text)
    if n == 0:
        print(
            f"\nWARNING: could not find a garbhaai.io/model-version annotation in "
            f"{path} -- manifest left unchanged. Check the Rollout's pod template "
            f"annotations block."
        )
        return False

    pattern_run = re.compile(r'(garbhaai\.io/model-run-id:\s*)"[^"]*"')
    new_text, n_run = pattern_run.subn(rf'\g<1>"{run_id}"', new_text)

    path.write_text(new_text)
    print(f"\nBumped {path} -> model-version={version!r}, model-run-id={run_id!r}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version", default=None,
        help="registered model version to consider promoting (default: highest registered version)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="alternative to --version: the MLflow run id whose registered version to consider",
    )
    parser.add_argument(
        "--approved-by", required=True,
        help="who is approving this promotion -- logged as a tag on the model version",
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST),
        help=f"Rollout manifest to bump on promotion (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run the gate and comparison, print the decision, change nothing",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if args.version:
        candidate = client.get_model_version(MODEL_REGISTRY_NAME, args.version)
    elif args.run_id:
        candidate = get_version_for_run(client, args.run_id, MODEL_REGISTRY_NAME)
        if candidate is None:
            raise SystemExit(
                f"no registered version of '{MODEL_REGISTRY_NAME}' found for run {args.run_id}"
            )
    else:
        candidate = get_highest_version(client, MODEL_REGISTRY_NAME)
        if candidate is None:
            raise SystemExit(f"no versions registered under '{MODEL_REGISTRY_NAME}'")

    print(f"Candidate: {MODEL_REGISTRY_NAME} version {candidate.version} (run {candidate.run_id})")

    structure_failures = check_structure(client, candidate.run_id)
    run = client.get_run(candidate.run_id)
    metrics = run.data.metrics
    completeness_failures = check_completeness(metrics)
    hard_failures = structure_failures + completeness_failures

    for f in hard_failures:
        print(f"FAIL: {f}")
    if hard_failures:
        print(f"\n{len(hard_failures)} hard failure(s) -- refusing to promote.")
        sys.exit(1)

    cand_mean_acc = metrics["mean_accuracy"]
    cand_worst_f1 = worst_class_f1(metrics)
    print(f"candidate: mean_accuracy={cand_mean_acc:.4f}, worst_class_f1={cand_worst_f1:.4f}")

    current_prod = get_current_production(client, MODEL_REGISTRY_NAME)
    prod_summary = None

    if current_prod is None:
        print(f"No version currently holds the '{PRODUCTION_ALIAS}' alias -- first promotion.")
        should_promote = True
    elif current_prod.version == candidate.version:
        print(f"version {candidate.version} already holds '{PRODUCTION_ALIAS}'. Nothing to do.")
        sys.exit(0)
    else:
        prod_run = client.get_run(current_prod.run_id)
        prod_metrics = prod_run.data.metrics
        prod_mean_acc = prod_metrics.get("mean_accuracy", float("nan"))
        has_all_f1 = all(f"{c}_f1" in prod_metrics for c in CLASS_NAMES)
        prod_worst_f1 = worst_class_f1(prod_metrics) if has_all_f1 else float("nan")
        print(
            f"current production (version {current_prod.version}): "
            f"mean_accuracy={prod_mean_acc:.4f}, worst_class_f1={prod_worst_f1:.4f}"
        )

        # The gate is on the weakest class, not the average -- a model that's
        # excellent at A/C but weak at B is not fit for use even if its
        # average looks fine. mean_accuracy only breaks a tie.
        if cand_worst_f1 > prod_worst_f1:
            should_promote = True
        elif cand_worst_f1 == prod_worst_f1 and cand_mean_acc > prod_mean_acc:
            should_promote = True
        else:
            should_promote = False
        prod_summary = {
            "version": current_prod.version,
            "mean_accuracy": prod_mean_acc,
            "worst_class_f1": prod_worst_f1,
        }

    if not should_promote:
        print(
            "\nCandidate does not beat current production on worst-class F1 "
            "(tie-break: mean_accuracy). Not promoting."
        )
        sys.exit(1)

    print(f"\nDecision: PROMOTE version {candidate.version} to '{PRODUCTION_ALIAS}'.")

    if args.dry_run:
        print("--dry-run set: registry and manifest left unchanged.")
        sys.exit(0)

    client.set_registered_model_alias(MODEL_REGISTRY_NAME, PRODUCTION_ALIAS, candidate.version)
    client.set_model_version_tag(MODEL_REGISTRY_NAME, candidate.version, "promoted_by", args.approved_by)
    client.set_model_version_tag(
        MODEL_REGISTRY_NAME, candidate.version, "promoted_at",
        datetime.now(timezone.utc).isoformat(),
    )
    if prod_summary:
        client.set_model_version_tag(
            MODEL_REGISTRY_NAME, candidate.version,
            "promoted_over_version", str(prod_summary["version"]),
        )
    print(
        f"Alias '{PRODUCTION_ALIAS}' now points to version {candidate.version} "
        f"(approved by {args.approved_by})."
    )

    bumped = bump_manifest(Path(args.manifest), candidate.version, candidate.run_id)
    if bumped:
        print(
            "\nThe registry alias has moved, but nothing is deployed until this change "
            "is committed and pushed -- Argo CD syncs from git, not from a local working "
            "tree. Run:\n"
            f"  git add {args.manifest}\n"
            f'  git commit -m "Promote {MODEL_REGISTRY_NAME} v{candidate.version} to production"\n'
            "  git push\n"
            "Argo CD will then sync the manifest change and Argo Rollouts will canary "
            "the new pods in (10% -> 100%)."
        )


if __name__ == "__main__":
    main()
