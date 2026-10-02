"""
Postgres and the feature-extraction HTTP call are both stubbed (via
dependency_overrides and monkeypatching httpx respectively) -- these
tests check this service's own logic: hashing, the day-support check,
the classifier-head math against the real bundled checkpoint, the
cached-review short-circuit, and the review upsert -- not Postgres or
another service's correctness.
"""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app, compute_image_hash, get_db_conn, load_checkpoint, predict_grade

client = TestClient(app)


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        sql_stripped = sql.strip()
        if sql_stripped.startswith("SELECT"):
            image_hash = params[0]
            row = self.store.get(image_hash)
            self._last_result = dict(row) if row else None
        elif sql_stripped.startswith("INSERT"):
            (image_hash, day, model_grade, model_confidence, probabilities_json,
             final_grade, doctor_overridden, patient_id, reviewed_at) = params
            self.store[image_hash] = {
                "image_hash": image_hash,
                "day": day,
                "model_grade": model_grade,
                "model_confidence": model_confidence,
                "model_probabilities": json.loads(probabilities_json),
                "final_grade": final_grade,
                "doctor_overridden": doctor_overridden,
                "patient_id": patient_id,
                "reviewed_at": reviewed_at,
            }

    def fetchone(self):
        return self._last_result


class FakeConn:
    def __init__(self, store=None):
        self.store = store if store is not None else {}
        self.committed = False

    def cursor(self, cursor_factory=None):
        return FakeCursor(self.store)

    def commit(self):
        self.committed = True


def _make_image_bytes(color=(50, 60, 70)):
    img = Image.new("RGB", (224, 224), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    # Phase 7 widened /health to also report which model is serving
    # (registry vs. bundled fallback -- see main.py's load_checkpoint()
    # docstring); this used to assert the pre-Phase-7 exact shape
    # ({"status": "ok"}) and would have failed the moment CI actually ran
    # a test against it. In CI there's no reachable MLflow registry, so
    # this always resolves to the bundled fallback -- asserted on the
    # stable parts of that shape rather than a model-version number that
    # has no reason to be any particular value here.
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["source"] in ("registry", "bundled_fallback")


def test_compute_image_hash_matches_ingestion_convention():
    b = _make_image_bytes()
    h = compute_image_hash(b)
    assert len(h) == 8
    assert h == compute_image_hash(b)


def test_checkpoint_loads_and_predict_grade_shape():
    checkpoint = load_checkpoint()
    assert checkpoint["class_names"] == ["A", "B", "C"]
    assert set(checkpoint["day_onehot"].keys()) == {"day3", "day4"}

    fake_features = np.random.RandomState(0).randn(512).tolist()
    result = predict_grade(fake_features, day=3)
    assert result["grade"] in ["A", "B", "C"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == {"A", "B", "C"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_grade_rejects_unsupported_day():
    fake_features = [0.0] * 512
    with pytest.raises(Exception):
        predict_grade(fake_features, day=5)


def test_grade_endpoint_returns_cached_review_without_calling_feature_extraction(monkeypatch):
    image_bytes = _make_image_bytes(color=(1, 2, 3))
    image_hash = compute_image_hash(image_bytes)
    store = {
        image_hash: {
            "image_hash": image_hash,
            "day": 3,
            "model_grade": "B",
            "model_confidence": 0.81,
            "model_probabilities": {"A": 0.1, "B": 0.81, "C": 0.09},
            "final_grade": "A",  # doctor overrode the model's B to A
            "doctor_overridden": True,
            "patient_id": "PATIENT_X",
            "reviewed_at": "2026-01-01T00:00:00+00:00",
        }
    }
    fake_conn = FakeConn(store)

    def fake_get_db_conn():
        yield fake_conn

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("feature-extraction should not be called for a cached review")

    monkeypatch.setattr("httpx.AsyncClient.post", fail_if_called)
    app.dependency_overrides[get_db_conn] = fake_get_db_conn
    try:
        resp = client.post(
            "/grade",
            data={"day": 3},
            files={"file": ("img.jpg", image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "cached_review"
        assert body["grade"] == "A"
        assert body["doctor_overridden"] is True
    finally:
        app.dependency_overrides.clear()


def test_review_upserts_record():
    fake_conn = FakeConn()

    def fake_get_db_conn():
        yield fake_conn

    app.dependency_overrides[get_db_conn] = fake_get_db_conn
    try:
        resp = client.post(
            "/review",
            data={
                "image_hash": "abcd1234",
                "day": 3,
                "model_grade": "B",
                "model_confidence": 0.7,
                "model_probabilities": json.dumps({"A": 0.2, "B": 0.7, "C": 0.1}),
                "final_grade": "B",
                "doctor_overridden": False,
                "patient_id": "PATIENT_Y",
            },
        )
        assert resp.status_code == 200
        assert fake_conn.store["abcd1234"]["final_grade"] == "B"
        assert fake_conn.committed is True
    finally:
        app.dependency_overrides.clear()
