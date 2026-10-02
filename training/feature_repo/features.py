"""
Phase 8 update: blur_score (a Phase 2 placeholder for a quality-check
output that was never actually wired into Feast) is dropped, and the
real 512-dim ResNet18 embedding is added -- this is what
services/inference/main.py now writes into the online store after every
live prediction (see that file's _push_to_feast()), and what
training/drift_engine/detect_drift.py reads back out to compare against
the training-set baseline.

services/inference/feast_schema.py keeps an intentionally duplicated
copy of this entity + feature view (the inference service stays
self-contained -- see its own docstring -- rather than importing this
module across a service boundary). Keep the two in sync by hand if this
schema changes again.
"""
from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Array, Float32, Int64

from entities import embryo_image

image_features_source = FileSource(
    path="data/image_features.parquet",
    timestamp_field="event_timestamp",
)

image_features_view = FeatureView(
    name="image_features",
    entities=[embryo_image],
    ttl=timedelta(days=365),
    schema=[
        Field(name="day", dtype=Int64),
        Field(name="model_confidence", dtype=Float32),
        Field(name="embedding", dtype=Array(Float32)),
    ],
    online=True,
    source=image_features_source,
)
