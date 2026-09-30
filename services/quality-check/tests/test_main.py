import io

from fastapi.testclient import TestClient
from PIL import Image

from main import app, check_image_quality

client = TestClient(app)


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
    resp = client.post(
        "/check-quality",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert any("under-exposed" in w for w in body["warnings"])
