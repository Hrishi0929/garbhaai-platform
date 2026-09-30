import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from main import app, preprocess_for_model

client = TestClient(app)


def _make_image_bytes(size=(300, 200), color=(10, 20, 30)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_preprocess_shape_and_normalization():
    tensor = preprocess_for_model(Image.new("RGB", (300, 200), (255, 255, 255)))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    # a pure white pixel, ImageNet-normalized, should land near (1 - mean) / std
    # per channel -- roughly 2.1-2.7 depending on channel, definitely not 0-1.
    assert tensor.max() > 1.5


def test_preprocess_endpoint_roundtrip():
    image_bytes = _make_image_bytes()
    resp = client.post(
        "/preprocess",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["shape"] == [3, 224, 224]
    raw = base64.b64decode(body["tensor_b64"])
    tensor = np.load(io.BytesIO(raw))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
