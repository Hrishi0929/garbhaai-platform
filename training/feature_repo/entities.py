from feast import Entity
from feast.value_type import ValueType

embryo_image = Entity(
    name="image_id",
    value_type=ValueType.STRING,
    description="Unique identifier for a single embryo image submitted for grading",
)
