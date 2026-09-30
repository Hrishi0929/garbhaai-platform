"""
Preprocessing service: turns a raw uploaded embryo image into the
normalized tensor the model expects.

Ported from garbhaai-embryo-grading/src/grading_core.py's
preprocess_for_model() -- must mirror src/preprocess.py's
load_and_preprocess_image() exactly (224x224 bilinear resize + ImageNet
mean/std normalization + HWC->CHW), since that's what the checkpoint
was trained against. Any drift here silently produces wrong grades, not
an error.

Returned as a base64-encoded .npy blob (via numpy.save into an in-memory
buffer) rather than a raw JSON array of floats, so shape/dtype survive
the round trip exactly and downstream services (feature-extraction) can
np.load() it back without any manual reshaping.
"""
import base64
import io

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="garbhaai-preprocessing")

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_model(pil_image: Image.Image) -> np.ndarray:
    """Returns a (3, 224, 224) float32 array, ImageNet-normalized, CHW."""
    resized = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))  # CHW
    return chw.astype(np.float32)


def encode_tensor(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    tensor = preprocess_for_model(pil_image)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "tensor_b64": encode_tensor(tensor),
    }
