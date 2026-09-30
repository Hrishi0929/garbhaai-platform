"""
Streamlit rendering itself isn't practically unit-testable without a
running browser, so these tests cover the pure helper logic in main.py
that has real behavior worth protecting: the review payload shape sent
to the inference service's /review endpoint.
"""
import json

from main import build_review_payload


def _fake_grade_result():
    return {
        "source": "model",
        "image_hash": "8bfa22b8",
        "grade": "B",
        "confidence": 0.81,
        "probabilities": {"A": 0.1, "B": 0.81, "C": 0.09},
    }


def test_build_review_payload_accept():
    payload = build_review_payload(
        _fake_grade_result(), day=3, patient_id="P1", final_grade="B", doctor_overridden=False
    )
    assert payload["image_hash"] == "8bfa22b8"
    assert payload["model_grade"] == "B"
    assert payload["final_grade"] == "B"
    assert payload["doctor_overridden"] is False
    assert json.loads(payload["model_probabilities"]) == {"A": 0.1, "B": 0.81, "C": 0.09}


def test_build_review_payload_override():
    payload = build_review_payload(
        _fake_grade_result(), day=3, patient_id="P1", final_grade="C", doctor_overridden=True
    )
    assert payload["model_grade"] == "B"  # what the model said
    assert payload["final_grade"] == "C"  # what the doctor corrected it to
    assert payload["doctor_overridden"] is True
