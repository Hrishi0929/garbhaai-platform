"""
These tests stub out Postgres and RustFS entirely via FastAPI's
dependency_overrides -- CI has no docker-compose stack available, and
these unit tests exist to check the service's own logic (hashing,
storage-key layout, request/response shape, the already_seen branch),
not to re-verify Postgres or boto3 themselves.
"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from main import app, build_storage_key, compute_image_hash, get_db_conn, get_storage_client


class FakeCursor:
    def __init__(self, existing_hashes):
        self.existing_hashes = existing_hashes
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        if sql.strip().startswith("SELECT 1"):
            image_hash = params[0]
            self._last_result = (1,) if image_hash in self.existing_hashes else None
        else:
            self._last_result = None

    def fetchone(self):
        return self._last_result


class FakeConn:
    def __init__(self, existing_hashes=()):
        self.existing_hashes = set(existing_hashes)
        self.committed = False

    def cursor(self):
        return FakeCursor(self.existing_hashes)

    def commit(self):
        self.committed = True


class FakeStorage:
    def __init__(self):
        self.put_calls = []

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)


def _make_image_bytes(color=(1, 2, 3)):
    img = Image.new("RGB", (50, 50), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_compute_image_hash_deterministic():
    b = _make_image_bytes()
    assert compute_image_hash(b) == compute_image_hash(b)
    assert len(compute_image_hash(b)) == 8


def test_build_storage_key():
    assert build_storage_key("abcd1234", 3) == "day3/abcd1234.jpg"


def test_ingest_new_image_stores_and_records():
    fake_conn = FakeConn()
    fake_storage = FakeStorage()

    def fake_get_db_conn():
        yield fake_conn

    app.dependency_overrides[get_db_conn] = fake_get_db_conn
    app.dependency_overrides[get_storage_client] = lambda: fake_storage
    try:
        client = TestClient(app)
        resp = client.post(
            "/ingest",
            data={"patient_id": "PATIENT_001", "day": 3},
            files={"file": ("img.jpg", _make_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["already_seen"] is False
        assert len(fake_storage.put_calls) == 1
        assert fake_conn.committed is True
    finally:
        app.dependency_overrides.clear()


def test_ingest_duplicate_image_skips_storage_write():
    image_bytes = _make_image_bytes(color=(9, 9, 9))
    image_hash = compute_image_hash(image_bytes)
    fake_conn = FakeConn(existing_hashes={image_hash})
    fake_storage = FakeStorage()

    def fake_get_db_conn():
        yield fake_conn

    app.dependency_overrides[get_db_conn] = fake_get_db_conn
    app.dependency_overrides[get_storage_client] = lambda: fake_storage
    try:
        client = TestClient(app)
        resp = client.post(
            "/ingest",
            data={"patient_id": "PATIENT_002", "day": 3},
            files={"file": ("img.jpg", image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["already_seen"] is True
        # duplicate hash -> object already in RustFS -> no redundant write
        assert len(fake_storage.put_calls) == 0
    finally:
        app.dependency_overrides.clear()
