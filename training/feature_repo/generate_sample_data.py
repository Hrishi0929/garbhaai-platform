"""
Generates a tiny placeholder parquet file so Feast has something to apply
and materialize against. This offline file is still only ever used to
satisfy FileSource's validation at `feast apply` time -- as of Phase 8,
real data reaches the online store exclusively through
services/inference/main.py's live write_to_online_store() calls after
each prediction, never through this file or `feast materialize`. This
script (and the old materialize-incremental flow in this folder's
README) is kept only as a way to smoke-test the Feast <-> Redis wiring
in isolation, without needing the full inference service running.

Run once: python generate_sample_data.py

Note: this file lives in the Feast repo directory, which `feast apply`
scans and imports to discover entities/feature views. Everything here
must stay inside `if __name__ == "__main__":` or it runs a second time,
unintentionally, during every `feast apply`.
"""
import pandas as pd
from datetime import datetime, timezone, timedelta


def main():
    now = datetime.now(timezone.utc)

    rows = []
    for i in range(10):
        rows.append({
            "image_id": f"GBR_PLACEHOLDER_{i:03d}",
            "event_timestamp": now - timedelta(minutes=10 - i),
            "day": 3 if i % 2 == 0 else 4,
            "model_confidence": 0.5 + 0.02 * i,  # stand-in for the last inference's confidence
            "embedding": [0.1 * i] * 512,         # stand-in for a real ResNet18 feature vector
        })

    df = pd.DataFrame(rows)
    df.to_parquet("data/image_features.parquet")
    print(f"wrote {len(df)} placeholder rows to data/image_features.parquet")


if __name__ == "__main__":
    main()
