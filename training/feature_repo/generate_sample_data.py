"""
Generates a tiny placeholder parquet file so Feast has something to apply
and materialize against before the real feature-extraction service
(Phase 4) exists. Columns are a stand-in for what that service will
eventually write: per-image features derived from the embryo photo plus
a timestamp Feast needs for point-in-time correctness.

Run once: python generate_sample_data.py
"""
import pandas as pd
from datetime import datetime, timedelta

now = datetime.utcnow()

rows = []
for i in range(10):
    rows.append({
        "image_id": f"GBR_PLACEHOLDER_{i:03d}",
        "event_timestamp": now - timedelta(minutes=10 - i),
        "day": 3 if i % 2 == 0 else 4,
        "blur_score": 0.1 * i,          # stand-in for the quality-check service's output
        "model_confidence": 0.5 + 0.02 * i,  # stand-in for the last inference's confidence
    })

df = pd.DataFrame(rows)
df.to_parquet("data/image_features.parquet")
print(f"wrote {len(df)} placeholder rows to data/image_features.parquet")
