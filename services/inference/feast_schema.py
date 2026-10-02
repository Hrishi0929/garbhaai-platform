"""
Entity + FeatureView definitions for the "image_features" view, mirroring
training/feature_repo/features.py exactly. Duplicated here (not imported
across the service boundary) to keep this service self-contained -- see
main.py's module docstring for why. Keep the two files in sync by hand if
the schema changes.

Unlike training/feature_repo (which drives `feast apply`/
`feast materialize` from the CLI against a feature_store.yaml file), this
module builds everything purely in Python: get_feast_store() constructs a
RepoConfig directly (no YAML file needed -- verified against the real,
installed feast==0.47.0 API) and calls store.apply() once at process
startup. That's what lets this run unmodified whether the service is on
the host, in Docker Compose, or in the kind cluster -- only the Redis
host/port env vars change.
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Array, Float32, Int64
from feast.value_type import ValueType

embryo_image = Entity(
    name="image_id",
    value_type=ValueType.STRING,
    description="Unique identifier for a single embryo image (compute_image_hash()'s output).",
)

# FileSource is required for FeatureView construction/validation even
# though this view is only ever written to directly via
# write_to_online_store() and only ever read back by detect_drift.py's
# get_online_features() calls -- the offline/batch path through this file
# is never actually used. _ensure_offline_source_exists() in main.py
# creates an empty file satisfying this at startup so `store.apply()`
# doesn't fail validating a path that doesn't exist yet.
OFFLINE_SOURCE_PATH = "feast_offline_placeholder.parquet"

image_features_source = FileSource(
    path=OFFLINE_SOURCE_PATH,
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
