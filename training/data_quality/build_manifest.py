"""
Phase 3: builds a flat manifest of the raw embryo image dataset so
Great Expectations has a tabular dataset to validate (image quality
checks, class balance, duplicate detection) instead of a folder of
images GE can't reason about directly.

This walks the Day3/Day4/Day5 dataset folders (same layout as
../../../Dataset -- Day 3/4/5 Dataset > Grade A/B/C > images), computes
an MD5 hash per file, and writes one row per image to a CSV.

Also emits an OpenLineage START/COMPLETE run event to Marquez, same as
ge_validate.py (duplicated, not imported -- these two scripts are each
meant to be runnable standalone per their own docstrings, matching this
repo's convention of self-contained scripts over shared helper
modules). Declaring manifest.csv as this job's *output* dataset, and
the same dataset as ge_validate.py's *input*, is what lets Marquez
chain the two job runs into one connected graph instead of two
unrelated floating boxes -- see ge_validate.py's own docstring for the
matching half of this.

Usage:
  python build_manifest.py                          # uses default path below
  python build_manifest.py --raw-root /some/other/path
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import OutputDataset, RunEvent, RunState, Run, Job
from openlineage.client.facet_v2 import (
    documentation_job,
    output_statistics_output_dataset,
    schema_dataset,
    source_code_location_job,
)
from openlineage.client.uuid import generate_new_uuid

MARQUEZ_URL = "http://localhost:5002"
JOB_NAMESPACE = "garbhaai"
JOB_NAME = "data_quality.build_raw_manifest"
REPO_URL = "https://github.com/Hrishi0929/garbhaai-platform"
SCRIPT_PATH = "training/data_quality/build_manifest.py"


def job_facets() -> dict:
    """Plain-language description + link to this script's source, shown on the
    job's page in Marquez."""
    return {
        "documentation": documentation_job.DocumentationJobFacet(
            description=(
                "Walks the Day 3/4/5 raw embryo image folders, computes an MD5 "
                "hash for every image, and writes one row per image (id, day, "
                "grade, path, size, hash) to manifest.csv for validation."
            )
        ),
        "sourceCodeLocation": source_code_location_job.SourceCodeLocationJobFacet(
            type="git",
            url=f"{REPO_URL}/blob/main/{SCRIPT_PATH}",
            repoUrl=REPO_URL,
            path=SCRIPT_PATH,
            branch="main",
        ),
    }


def manifest_dataset(df: pd.DataFrame, size_bytes: int = None) -> OutputDataset:
    """The manifest Dataset, now carrying a schema facet (one field per CSV
    column) so Marquez's dataset page shows real columns instead of
    "0 columns". Same namespace + name as before, so it's still the same
    dataset node -- both build_manifest.py and ge_validate.py build it the
    same way (duplicated, not imported, per this repo's self-contained-
    scripts convention) so the shared node's schema stays consistent."""
    def _ol_type(dtype) -> str:
        return "INTEGER" if pd.api.types.is_integer_dtype(dtype) else "STRING"

    fields = [
        schema_dataset.SchemaDatasetFacetFields(name=col, type=_ol_type(dtype))
        for col, dtype in df.dtypes.items()
    ]
    return OutputDataset(
        namespace=JOB_NAMESPACE,
        name="training.manifest_csv",
        facets={"schema": schema_dataset.SchemaDatasetFacet(fields=fields)},
        outputFacets={
            "outputStatistics": output_statistics_output_dataset.OutputStatisticsOutputDatasetFacet(
                rowCount=len(df), size=size_bytes
            )
        },
    )

DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[3] / "Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def emit_lineage_event(ol_client, run_id, event_time, state, outputs=None):
    ol_client.emit(
        RunEvent(
            eventType=state,
            eventTime=event_time.isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=JOB_NAMESPACE, name=JOB_NAME, facets=job_facets()),
            producer="https://github.com/Hrishi0929/garbhaai-platform/training/data_quality",
            outputs=outputs or [],
        )
    )


def build_manifest(raw_root: Path) -> pd.DataFrame:
    rows = []
    for day_dir in sorted(raw_root.glob("Day * Dataset")):
        day = int(day_dir.name.split()[1])
        for grade_dir in sorted(day_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            grade = grade_dir.name.replace("Grade ", "").strip()
            for f in sorted(grade_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                md5 = hashlib.md5(f.read_bytes()).hexdigest()
                rows.append({
                    "image_id": f.stem,
                    "day": day,
                    "grade": grade,
                    "file_path": str(f),
                    "file_size_bytes": f.stat().st_size,
                    "md5_hash": md5,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=Path("manifest.csv"))
    args = parser.parse_args()

    if not args.raw_root.exists():
        raise SystemExit(f"raw dataset root not found: {args.raw_root}")

    ol_client = OpenLineageClient(url=MARQUEZ_URL)
    run_id = str(generate_new_uuid())
    emit_lineage_event(ol_client, run_id, datetime.now(timezone.utc), RunState.START)

    df = build_manifest(args.raw_root)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(df["grade"].value_counts())
    print(f"duplicate md5 hashes: {df['md5_hash'].duplicated().sum()}")

    emit_lineage_event(
        ol_client, run_id, datetime.now(timezone.utc), RunState.COMPLETE,
        outputs=[manifest_dataset(df, size_bytes=args.out.stat().st_size)],
    )
    print(f"\nOpenLineage run {run_id} sent to Marquez at {MARQUEZ_URL} "
          f"(namespace={JOB_NAMESPACE}, job={JOB_NAME}) -- check http://localhost:3000")


if __name__ == "__main__":
    main()
