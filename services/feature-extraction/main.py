"""
Feature-extraction service: runs the frozen ResNet18 backbone (only --
not the day-conditioned classifier head) over a preprocessed embryo
image and returns the 512-dim feature vector.

This is a genuinely new service, not a direct port: the original
garbhaai-embryo-grading app fused backbone + classifier head into one
DayConditionedClassifier module and only ever exposed the final grade.
Splitting the backbone out means the 512-dim feature vector can be
stored in Feast (training/feature_repo) as real data instead of the
Phase 2 placeholder floats, and reused by anything else that wants
image embeddings without re-running the full classifier.

Accepts a raw image directly (not a pre-computed tensor from the
preprocessing service) and preprocesses it internally -- see the
services/README note on why each service stays self-contained rather
than importing a shared library.

The classifier head (day-conditioned Linear layer -> grade + confidence)
stays out of this service on purpose: that belongs to the Phase 5
"inference" service, along with the doctor-review flow and the
labeled-data cache. This service's job stops at "here is the feature
vector for this image."
"""
import io
import os

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision.models import resnet18

app = FastAPI(title="garbhaai-feature-extraction")

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_backbone_cache = {}


def preprocess_for_model(pil_image: Image.Image) -> np.ndarray:
    """Same normalization as services/preprocessing -- must match what
    the backbone was trained against."""
    resized = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def load_backbone() -> nn.Module:
    """Frozen ResNet18 with its classification head removed (Identity),
    so forward() returns the 512-dim pooled feature vector instead of
    1000-way ImageNet logits. Cached at module level -- loading ImageNet
    weights on every request would be needlessly slow.

    Weight source is env-controlled (GARBHAAI_RESNET_WEIGHTS, default
    "IMAGENET1K_V1") so tests can force random init and avoid a network
    fetch from download.pytorch.org -- shape and determinism tests don't
    depend on which weights are loaded, only production inference does."""
    if "backbone" not in _backbone_cache:
        weights_name = os.environ.get("GARBHAAI_RESNET_WEIGHTS", "IMAGENET1K_V1")
        model = resnet18(weights=weights_name or None)
        model.fc = nn.Identity()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _backbone_cache["backbone"] = model
    return _backbone_cache["backbone"]


def extract_features(pil_image: Image.Image) -> np.ndarray:
    tensor = preprocess_for_model(pil_image)
    batch = torch.from_numpy(tensor).unsqueeze(0)  # (1, 3, 224, 224)
    backbone = load_backbone()
    with torch.no_grad():
        features = backbone(batch)  # (1, 512)
    return features.squeeze(0).numpy()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-features")
async def extract_features_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    features = extract_features(pil_image)
    return {
        "feature_dim": int(features.shape[0]),
        "features": features.tolist(),
    }
