#!/usr/bin/env python3
"""
Phase 6 training engine: unified Day3+Day4 day-conditioned classifier,
logged to the self-hosted MLflow server (infra/local, port 5500) instead
of the local sqlite store the pre-platform version used.

Ported from garbhaai-embryo-grading/src/train_unified.py -- same model
(DayConditionedClassifier: frozen ResNet18 backbone + 2-dim day one-hot
-> Linear(514, 3)), same LOOCV evaluation strategy, same "train a final
model on all pooled data, that's the deployable artifact" structure, same
checkpoint format (verified against the real checkpoint in Phase 5 --
see services/inference/main.py's docstring).

The one real change: instead of recomputing the ResNet18 backbone forward
pass from raw images on every fold (what the old script did), this calls
the Phase 4 feature-extraction service ONCE per image up front and trains
only the classifier head against the cached 512-dim vectors. Functionally
identical (the backbone is frozen either way, so its output per image
never changes across folds) and faster, but more importantly: training
now runs through the exact same feature-extraction code path that
production inference uses, rather than a second copy of the backbone
logic living in the training script. If feature-extraction's
preprocessing/backbone ever changes, training automatically sees the
same change inference does.

Day 5 is intentionally excluded -- see the Phase 3 data-quality findings
(different grading taxonomy) and the original train_unified.py docstring.

Requires: the Phase 4 services (at least feature-extraction) and the
Phase 6 MLflow server both reachable. Usage:
  cd infra/local && ./bootstrap.sh          # mlflow, postgres, rustfs, redis
  cd ../../services && docker compose up -d  # feature-extraction, etc.
  cd ../training/training_engine
  pip install -r requirements.txt
  python train.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut

REPO_ROOT = Path(__file__).resolve().parents[2]  # garbhaai-platform/
DATASET_ROOT = Path(os.environ.get("GARBHAAI_DATASET_ROOT", REPO_ROOT.parent / "Dataset"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEATURE_EXTRACTION_URL = os.environ.get("FEATURE_EXTRACTION_URL", "http://localhost:8004")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500")
MLFLOW_EXPERIMENT = "garbhaai_embryo_grading"
MODEL_REGISTRY_NAME = "garbhaai-day-classifier"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["A", "B", "C"]
STAGES = ["day3", "day4"]  # day5 intentionally excluded -- different grading taxonomy
DAY_DIR_NAME = {"day3": "Day 3 Dataset", "day4": "Day 4 Dataset"}
DAY_ONEHOT = {"day3": [1.0, 0.0], "day4": [0.0, 1.0]}
N_DAY_DIMS = 2
EPOCHS_PER_FOLD = 15
LR = 1e-3


def extract_features(image_path: Path) -> list:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{FEATURE_EXTRACTION_URL}/extract-features",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["features"]


def load_stage(stage: str):
    """Walks Dataset/Day N Dataset/Grade X/*.jpg and calls the
    feature-extraction service for each image -- mirrors
    training/data_quality/build_manifest.py's folder-walking logic."""
    stage_dir = DATASET_ROOT / DAY_DIR_NAME[stage]
    if not stage_dir.exists():
        raise SystemExit(f"dataset folder not found: {stage_dir}")

    features, labels, image_ids = [], [], []
    for grade_dir in sorted(stage_dir.iterdir()):
        if not grade_dir.is_dir():
            continue
        grade = grade_dir.name.replace("Grade ", "").strip()
        if grade not in CLASS_NAMES:
            continue  # defensive -- day3/day4 folders are plain A/B/C only
        label = CLASS_NAMES.index(grade)
        for f in sorted(grade_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            features.append(extract_features(f))
            labels.append(label)
            image_ids.append(f.stem)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    day_onehot = np.tile(DAY_ONEHOT[stage], (X.shape[0], 1)).astype(np.float32)
    return X, y, day_onehot, image_ids


def run_fold(X_train, Xday_train, y_train, X_val, Xday_val):
    """Trains just the classifier head (Linear(514, 3)) against
    precomputed features -- the backbone is frozen and lives entirely in
    the feature-extraction service, so there's nothing else to train."""
    head = nn.Linear(X_train.shape[1] + N_DAY_DIMS, len(CLASS_NAMES))
    Xt = torch.tensor(np.concatenate([X_train, Xday_train], axis=1), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(np.concatenate([X_val, Xday_val], axis=1), dtype=torch.float32)

    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    head.train()
    for _epoch in range(EPOCHS_PER_FOLD):
        optimizer.zero_grad()
        loss = criterion(head(Xt), yt)
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        val_preds = head(Xv).argmax(dim=1).numpy()
    return val_preds, float(loss.item())


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_parts, y_parts, day_parts, ids_parts, stage_label_parts = [], [], [], [], []
    for stage in STAGES:
        print(f"Extracting features for {stage}...")
        X, y, day_onehot, image_ids = load_stage(stage)
        X_parts.append(X)
        y_parts.append(y)
        day_parts.append(day_onehot)
        ids_parts.extend(image_ids)
        stage_label_parts.extend([stage] * X.shape[0])
        print(f"  {stage}: {X.shape[0]} images")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X_day = np.concatenate(day_parts, axis=0)
    image_ids = ids_parts
    stage_labels = stage_label_parts

    class_counts = [int((y == i).sum()) for i in range(3)]
    print(f"Pooled {STAGES}: X={X.shape}, y={y.shape}, class counts = {class_counts}")

    run_name = f"unified_day3day4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with mlflow.start_run(run_name=run_name) as run:
        loo = LeaveOneOut()
        n_splits = loo.get_n_splits(X)

        mlflow.log_params({
            "stages_included": ",".join(STAGES),
            "stage_excluded": "day5 (different grading taxonomy -- see Phase 3 findings)",
            "day_conditioning": "one-hot concat before final linear layer",
            "n_day_dims": N_DAY_DIMS,
            "cv_strategy": "leave_one_out",
            "n_folds": n_splits,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "lr": LR,
            "backbone": "resnet18",
            "feature_source": "feature-extraction service (Phase 4)",
            "n_images": int(X.shape[0]),
        })

        all_true, all_pred, fold_accs = [], [], []
        final_train_loss = None
        for fold_i, (train_idx, val_idx) in enumerate(loo.split(X)):
            preds, final_train_loss = run_fold(
                X[train_idx], X_day[train_idx], y[train_idx], X[val_idx], X_day[val_idx]
            )
            acc = accuracy_score(y[val_idx], preds)
            fold_accs.append(acc)
            all_true.extend(y[val_idx].tolist())
            all_pred.extend(preds.tolist())
            val_desc = [f"{image_ids[i]}({stage_labels[i]})" for i in val_idx]
            print(f"  fold {fold_i + 1}/{n_splits}: val_acc={acc:.3f} (val images: {val_desc})")
            mlflow.log_metric("fold_val_accuracy", acc, step=fold_i)

        precision, recall, f1, support = precision_recall_fscore_support(
            all_true, all_pred, labels=[0, 1, 2], zero_division=0
        )
        cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
        mean_acc = float(np.mean(fold_accs))

        results = {
            "model": "unified_day3_day4",
            "stages_included": STAGES,
            "n_images": int(X.shape[0]),
            "cv_strategy": "leave_one_out",
            "n_folds": n_splits,
            "fold_accuracies": [round(a, 4) for a in fold_accs],
            "mean_accuracy": round(mean_acc, 4),
            "per_class": {
                CLASS_NAMES[i]: {
                    "precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(support[i]),
                }
                for i in range(3)
            },
            "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm.tolist()},
            "caveat": (
                f"Leave-one-out CV across all {int(X.shape[0])} pooled images "
                f"(class counts: {class_counts}) -- every image was used as the "
                f"held-out validation example exactly once, the most "
                f"data-efficient honest estimate available at this dataset "
                f"size. Still a small-sample estimate with real variance: "
                f"treat the accuracy number as directional, not precise."
            ),
        }

        mlflow.log_metrics({"mean_accuracy": mean_acc, "final_train_loss": final_train_loss})
        for cls_name, cls_metrics in results["per_class"].items():
            for metric_name, val in cls_metrics.items():
                mlflow.log_metric(f"{cls_name}_{metric_name}", val)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = OUTPUT_DIR / "unified_day3_day4_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2) + "\n")
        mlflow.log_artifact(str(metrics_path))

        print(json.dumps(results, indent=2))

        # --- train the deployable model on ALL pooled data (none held out) ---
        final_head = nn.Linear(X.shape[1] + N_DAY_DIMS, len(CLASS_NAMES))
        optimizer = torch.optim.Adam(final_head.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        Xt = torch.tensor(np.concatenate([X, X_day], axis=1), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        final_head.train()
        for _epoch in range(EPOCHS_PER_FOLD):
            optimizer.zero_grad()
            loss = criterion(final_head(Xt), yt)
            loss.backward()
            optimizer.step()
        final_head.eval()

        checkpoint_path = OUTPUT_DIR / "unified_day3_day4_classifier.pt"
        torch.save({
            "classifier_state_dict": final_head.state_dict(),
            "pretrained_backbone": True,
            "class_names": CLASS_NAMES,
            "day_onehot": DAY_ONEHOT,
            "backbone_arch": "resnet18",
            "trained_on_n_images": int(X.shape[0]),
        }, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))

        registered = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/{checkpoint_path.name}",
            name=MODEL_REGISTRY_NAME,
        )
        print(f"\nSaved deployable checkpoint -> {checkpoint_path}")
        print(f"MLflow run: {run.info.run_id} (experiment '{MLFLOW_EXPERIMENT}')")
        print(f"Registered as {MODEL_REGISTRY_NAME} version {registered.version}")
        print(f"View at {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
