import io
import os

# Force random-init weights for tests -- must be set before `main` is
# imported (load_backbone() reads it once and caches the model), so
# tests never depend on reaching download.pytorch.org for real
# ImageNet weights.
os.environ["GARBHAAI_RESNET_WEIGHTS"] = ""

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from main import app, extract_features  # noqa: E402

client = TestClient(app)


def _make_image_bytes(size=(300, 200), color=(10, 20, 30)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_extract_features_shape():
    features = extract_features(Image.new("RGB", (224, 224), (128, 128, 128)))
    assert features.shape == (512,)


def test_extract_features_endpoint():
    resp = client.post(
        "/extract-features",
        files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_dim"] == 512
    assert len(body["features"]) == 512


def test_same_image_same_features():
    # a frozen backbone in eval() mode is deterministic -- identical
    # input must produce bit-identical output, which is the whole
    # premise of caching/reusing extracted features downstream.
    img = Image.new("RGB", (224, 224), (77, 88, 99))
    f1 = extract_features(img)
    f2 = extract_features(img)
    assert (f1 == f2).all()
