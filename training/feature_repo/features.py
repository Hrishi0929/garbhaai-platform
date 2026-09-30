from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

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
        Field(name="blur_score", dtype=Float32),
        Field(name="model_confidence", dtype=Float32),
    ],
    online=True,
    source=image_features_source,
)
