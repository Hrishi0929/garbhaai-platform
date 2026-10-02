"""
The Postgres write this service now does (see main.py's Phase 9
gap-closure docstring) is stubbed out via FastAPI's dependency_overrides,
same pattern as ingestion/inference -- CI has no docker-compose stack
available, and these unit tests exist to check the service's own logic
(blur/exposure detection, the quality_checks row getting written with
the right fields), not to re-verify Postgres itself.
"""
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from main import app, check_image_quality, compute_image_hash, get_db_conn

client = TestClient(app)


class FakeCursor:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        sql_stripped = sql.strip()
        if sql_stripped.startswith("INSERT"):
            image_hash, ok, warnings_json, checked_at = params
            self.store.append(
                {
                    "image_hash": image_hash,
                    "ok": ok,
                    "warnings": json.loads(warnings_json),
                    "checked_at": checked_at,
                }
            )


class FakeConn:
    def __init__(self):
        self.rows = []
        self.committed = False

    def cursor(self):
        return FakeCursor(self.rows)

    def commit(self):
        self.committed = True


def _make_image_bytes(mode="RGB", size=(224, 224), color=(128, 128, 128)):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_flat_gray_image_flagged_blurry():
    # a perfectly flat image has zero laplacian variance -- always below
    # the blur threshold, so this is a cheap, deterministic way to
    # exercise the "blurred" warning path without needing a real photo.
    result = check_image_quality(Image.new("RGB", (224, 224), (128, 128, 128)))
    assert result["ok"] is True
    assert any("blurred" in w for w in result["warnings"])


def test_underexposed_image_flagged():
    result = check_image_quality(Image.new("RGB", (224, 224), (0, 0, 0)))
    assert any("under-exposed" in w for w in result["warnings"])


def test_check_quality_endpoint():
    image_bytes = _make_image_bytes(color=(0, 0, 0))
    fake_conn = FakeConn()

    def fake_get_db_conn():
        yield fake_conn

    app.dependency_overrides[get_db_conn] = fake_get_db_conn
    try:
        resp = client.post(
            "/check-quality",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert any("under-exposed" in w for w in body["warnings"])
    finally:
        app.dependency_overrides.clear()


def test_check_quality_endpoint_writes_quality_checks_row():
    image_bytes = _make_image_bytes(color=(0, 0, 0))
    expected_hash = compute_image_hash(image_bytes)
    fake_conn = FakeConn()

    def fake_get_db_conn():
        yield fake_conn

    app.dependency_overrides[get_db_conn] = fake_get_db_conn
    try:
        resp = client.post(
            "/check-quality",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        assert fake_conn.committed is True
        assert len(fake_conn.rows) == 1
        row = fake_conn.rows[0]
        assert row["image_hash"] == expected_hash
        assert row["ok"] is True
        assert any("under-exposed" in w for w in row["warnings"])
    finally:
        app.dependency_overrides.clear()
