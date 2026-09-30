# GarbhaAI data quality & lineage (Phase 3)

Two things live here:

1. **Great Expectations** validates the raw dataset against a set of
   rules (no nulls in `image_id`, grades restricted to A/B/C, days
   restricted to 3/4/5, no duplicate images by content hash, a sane
   row count). This is a dataset-level quality gate -- distinct from
   the per-upload blur/quality check the live UI does on a single
   image at grading time.
2. **OpenLineage -> Marquez** records that this validation job ran,
   when, and whether it passed, so there's an auditable lineage trail
   of every time the dataset was checked.

## Setup

```
pip install "great_expectations>=1.0" "openlineage-python"
cd infra/local/marquez && docker compose up -d && cd -
cd training/data_quality
python build_manifest.py        # writes manifest.csv from the raw Dataset folder
python ge_validate.py           # runs GE checks + emits lineage events to Marquez (posts to http://localhost:5002)
```

Then open http://localhost:3000 and look for the `garbhaai` namespace
and the `data_quality.validate_raw_manifest` job -- you should see a
run with a COMPLETE or FAIL state and a timestamp matching when you
ran it.

## Expected result right now

The `unique md5_hash` expectation is expected to **fail** on the
current dataset: there's a known byte-identical duplicate image in
`Day 3 Dataset / Grade B` (found earlier by hand with an ad hoc hash
script). This is the intended outcome -- it demonstrates the gate
catching a real, previously-manual-only data-quality finding
automatically. Fixing the underlying duplicate is a separate,
optional cleanup; the point of this phase is that the gate exists and
catches it, not that the dataset is already clean.

## What happens later

`build_manifest.py`'s default `--raw-root` points at the
`Dataset/` folder as it exists today. Once the real ingestion service
(Phase 4) is writing incoming images somewhere else (RustFS via the
object-storage service), point `--raw-root` there, or replace the
manifest step with a query against Postgres if ingested-image
metadata ends up there instead.
