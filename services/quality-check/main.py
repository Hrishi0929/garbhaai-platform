"""
Quality-check service: blur/exposure/integrity check on an uploaded
embryo image, run before it's sent anywhere else in the pipeline.

Ported from garbhaai-embryo-grading/src/grading_core.py's
check_image_quality(), unchanged in logic -- this service just wraps it
behind an HTTP API instead of a direct in-process import, so ingestion,
the UI, or anything else can call it without needing PyTorch installed
(this service has no model dependency at all).
"""
import io

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from scipy.ndimage import laplace

app = FastAPI(title="garbhaai-quality-check")

# NOT calibrated against this dataset -- there were no known-blurry example
# images to tune against when this was written. If real blurry uploads slip
# through (or good images get flagged), adjust this after seeing a few.
BLUR_VARIANCE_THRESHOLD = 100.0
UNDEREXPOSED_MEAN_THRESHOLD = 20.0   # 0-255 grayscale scale
OVEREXPOSED_MEAN_THRESHOLD = 235.0


def check_image_quality(pil_image: Image.Image) -> dict:
    """Returns {"ok": bool, "warnings": [str, ...]}. "ok" is False only
    for a genuinely unreadable file; blur/exposure are warnings the
    caller (UI) can choose to let the user proceed past, not hard
    failures."""
    warnings = []
    try:
        gray = np.asarray(pil_image.convert("L"), dtype=np.float32)
    except Exception as e:
        return {"ok": False, "warnings": [f"Could not read this image file ({e})."]}

    if gray.size == 0:
        return {"ok": False, "warnings": ["Image appears to be empty/corrupt."]}

    blur_variance = float(np.var(laplace(gray)))
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        warnings.append(
            f"Image looks blurred (sharpness score {blur_variance:.0f}, "
            f"below the {BLUR_VARIANCE_THRESHOLD:.0f} threshold)."
        )

    mean_brightness = float(gray.mean())
    if mean_brightness < UNDEREXPOSED_MEAN_THRESHOLD:
        warnings.append(
            f"Image looks under-exposed / too dark (mean brightness {mean_brightness:.0f})."
        )
    elif mean_brightness > OVEREXPOSED_MEAN_THRESHOLD:
        warnings.append(
            f"Image looks over-exposed / too bright (mean brightness {mean_brightness:.0f})."
        )

    return {"ok": True, "warnings": warnings}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-quality")
async def check_quality(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    return check_image_quality(pil_image)
