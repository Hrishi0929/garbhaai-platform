from feast import Entity
from feast.types import String

embryo_image = Entity(
    name="image_id",
    value_type=String,
    description="Unique identifier for a single embryo image submitted for grading",
)
