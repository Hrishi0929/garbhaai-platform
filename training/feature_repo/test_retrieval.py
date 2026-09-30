"""
Phase 2 exit-gate check: after `feast apply` + `feast materialize-incremental`,
this confirms Redis actually has the features and Feast can serve them back.

Run: python test_retrieval.py
"""
from feast import FeatureStore

store = FeatureStore(repo_path=".")

result = store.get_online_features(
    features=[
        "image_features:day",
        "image_features:blur_score",
        "image_features:model_confidence",
    ],
    entity_rows=[{"image_id": "GBR_PLACEHOLDER_005"}],
).to_dict()

print(result)
assert result["model_confidence"][0] is not None, "Feature retrieval returned None -- materialization likely didn't run"
print("\nOK -- Feast served a real feature value back from Redis.")
