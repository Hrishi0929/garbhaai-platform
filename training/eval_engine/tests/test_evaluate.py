import math
from pathlib import Path

import torch

from evaluate import check_completeness, validate_checkpoint

REAL_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3]
    / "services" / "inference" / "model" / "unified_day3_day4_classifier.pt"
)


def _good_metrics():
    return {
        "mean_accuracy": 0.4,
        "final_train_loss": 0.1,
        "A_precision": 0.5, "A_recall": 0.5, "A_f1": 0.5,
        "B_precision": 0.3, "B_recall": 0.3, "B_f1": 0.3,
        "C_precision": 0.4, "C_recall": 0.4, "C_f1": 0.4,
    }


def test_check_completeness_passes_on_full_metrics():
    assert check_completeness(_good_metrics()) == []


def test_check_completeness_flags_missing_metric():
    metrics = _good_metrics()
    del metrics["mean_accuracy"]
    failures = check_completeness(metrics)
    assert any("mean_accuracy" in f for f in failures)


def test_check_completeness_flags_nan():
    metrics = _good_metrics()
    metrics["A_f1"] = math.nan
    failures = check_completeness(metrics)
    assert any("A_f1" in f and "NaN" in f for f in failures)


def test_validate_checkpoint_accepts_well_formed_dict():
    ckpt = {
        "class_names": ["A", "B", "C"],
        "day_onehot": {"day3": [1.0, 0.0], "day4": [0.0, 1.0]},
        "classifier_state_dict": {"weight": torch.zeros(3, 514), "bias": torch.zeros(3)},
    }
    assert validate_checkpoint(ckpt) == []


def test_validate_checkpoint_rejects_wrong_shape():
    ckpt = {
        "class_names": ["A", "B", "C"],
        "day_onehot": {"day3": [1.0, 0.0], "day4": [0.0, 1.0]},
        # missing the 2-dim day one-hot in the input size
        "classifier_state_dict": {"weight": torch.zeros(3, 512), "bias": torch.zeros(3)},
    }
    failures = validate_checkpoint(ckpt)
    assert any("shape" in f for f in failures)


def test_validate_checkpoint_rejects_wrong_class_names():
    ckpt = {
        "class_names": ["good", "bad"],
        "day_onehot": {"day3": [1.0, 0.0], "day4": [0.0, 1.0]},
        "classifier_state_dict": {"weight": torch.zeros(3, 514), "bias": torch.zeros(3)},
    }
    failures = validate_checkpoint(ckpt)
    assert any("class_names" in f for f in failures)


def test_validate_checkpoint_against_real_deployed_artifact():
    """The gate this eval engine implements should pass the exact
    checkpoint that's actually bundled into services/inference -- if it
    didn't, the gate would be rejecting production's own model."""
    if not REAL_CHECKPOINT_PATH.exists():
        import pytest
        pytest.skip(f"real checkpoint not found at {REAL_CHECKPOINT_PATH}")
    ckpt = torch.load(REAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert validate_checkpoint(ckpt) == []
