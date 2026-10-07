"""
Phase 3 exit-gate check: runs a Great Expectations validation suite
against the manifest built by build_manifest.py, then emits an
OpenLineage START/COMPLETE run event to Marquez so the run shows up
in Marquez's lineage graph. The COMPLETE/FAIL event also carries each
expectation's pass/fail as a dataQualityAssertions facet, so Marquez's
Quality badge on training.manifest_csv reflects the real check results.

Usage:
  python build_manifest.py       # produces manifest.csv first
  python ge_validate.py

This is deliberately checking for the real duplicate-image issue found
in the raw dataset (several byte-identical images across Day 3/4/5,
including one image labeled with two different grades) -- so the
"unique md5_hash" expectation is EXPECTED to fail on the current
dataset. That failure is the point: this is GE catching, in an
automated gate, data-quality bugs that were previously only found by
hand, one at a time.

The "grade" column's allowed value set is day-specific, not one flat
A/B/C: Day 3 uses cleavage-stage grading (A/B/C), Day 4 uses Morula
grading, Day 5 uses Blastocyst grading -- see the Nexus Vol. 11
"Embryo Grading" reference (IFS/Origio, 2019), which documents these
as three distinct, day-specific assessment schemes rather than
interchangeable labels. GRADE_VALUE_SET below lists every valid label
across all three days.

Declares manifest.csv (training.manifest_csv) as this job's *input*
dataset -- the same dataset build_manifest.py declares as its *output*
-- so Marquez chains the two job runs into one connected graph
(build job -> dataset -> this job) instead of two unrelated floating
boxes. See build_manifest.py's docstring for the matching half.
"""
import sys
from datetime import datetime, timezone

import pandas as pd
import great_expectations as gx

from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import InputDataset, RunEvent, RunState, Run, Job
from openlineage.client.facet_v2 import (
    data_quality_assertions_dataset,
    documentation_job,
    schema_dataset,
    source_code_location_job,
)
from openlineage.client.uuid import generate_new_uuid

MARQUEZ_URL = "http://localhost:5002"
JOB_NAMESPACE = "garbhaai"
JOB_NAME = "data_quality.validate_raw_manifest"
REPO_URL = "https://github.com/Hrishi0929/garbhaai-platform"
SCRIPT_PATH = "training/data_quality/ge_validate.py"


def job_facets() -> dict:
    """Plain-language description + link to this script's source, shown on the
    job's page in Marquez so a non-technical viewer can see what it does and
    where the code lives."""
    return {
        "documentation": documentation_job.DocumentationJobFacet(
            description=(
                "Quality gate for the raw-image manifest: runs Great Expectations "
                "checks (no missing IDs, valid grade labels, valid day, unique "
                "MD5 hash per image, sane row count). Fails the run if any check "
                "fails, e.g. when two images are byte-identical duplicates."
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


def manifest_dataset(df: pd.DataFrame, assertions=None) -> InputDataset:
    """The manifest Dataset, now carrying a schema facet (one field per CSV
    column) so Marquez's dataset page shows real columns instead of
    "0 columns". Same namespace + name as before, so it's still the same
    dataset node -- both build_manifest.py and ge_validate.py build it the
    same way (duplicated, not imported, per this repo's self-contained-
    scripts convention) so the shared node's schema stays consistent.

    When `assertions` is given (the per-check Great Expectations results),
    they ride along as a dataQualityAssertions input facet -- that is what
    fills in the Quality badge on the dataset page in Marquez and shows
    which exact check failed on which column."""
    def _ol_type(dtype) -> str:
        return "INTEGER" if pd.api.types.is_integer_dtype(dtype) else "STRING"

    fields = [
        schema_dataset.SchemaDatasetFacetFields(name=col, type=_ol_type(dtype))
        for col, dtype in df.dtypes.items()
    ]
    input_facets = {}
    if assertions:
        input_facets["dataQualityAssertions"] = (
            data_quality_assertions_dataset.DataQualityAssertionsDatasetFacet(
                assertions=assertions
            )
        )
    return InputDataset(
        namespace=JOB_NAMESPACE,
        name="training.manifest_csv",
        facets={"schema": schema_dataset.SchemaDatasetFacet(fields=fields)},
        inputFacets=input_facets,
    )

# Every valid grade label across all three day-specific assessment schemes
# (see module docstring). Day 3 = cleavage stage, Day 4 = Morula, Day 5 =
# Blastocyst -- these are not interchangeable, so this is a flat allow-list
# of the union, not a claim that any grade is valid on any day.
GRADE_VALUE_SET = [
    "A", "B", "C",
    "Morula A", "Morula B", "Morula C",
    "Blastocyst A", "Blastocyst B", "Blastocyst C",
]


def emit_lineage_event(ol_client, run_id, event_time, state, inputs=None):
    ol_client.emit(
        RunEvent(
            eventType=state,
            eventTime=event_time.isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=JOB_NAMESPACE, name=JOB_NAME, facets=job_facets()),
            producer="https://github.com/Hrishi0929/garbhaai-platform/training/data_quality",
            inputs=inputs or [],
        )
    )


def main():
    df = pd.read_csv("manifest.csv")

    ol_client = OpenLineageClient(url=MARQUEZ_URL)
    run_id = str(generate_new_uuid())
    emit_lineage_event(
        ol_client, run_id, datetime.now(timezone.utc), RunState.START,
        inputs=[manifest_dataset(df)],
    )

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("manifest_source")
    data_asset = data_source.add_dataframe_asset("manifest")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_manifest")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="image_id"),
        gx.expectations.ExpectColumnValuesToBeInSet(column="grade", value_set=GRADE_VALUE_SET),
        gx.expectations.ExpectColumnValuesToBeInSet(column="day", value_set=[3, 4, 5]),
        gx.expectations.ExpectColumnValuesToBeUnique(column="md5_hash"),
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, max_value=10000),
    ]

    results = [batch.validate(exp) for exp in expectations]
    all_success = all(r.success for r in results)

    print("\n--- Great Expectations results ---")
    for exp, r in zip(expectations, results):
        status = "PASS" if r.success else "FAIL"
        print(f"[{status}] {exp.expectation_type} (column={getattr(exp, 'column', None)})")
        if not r.success:
            print(f"       {r.result}")

    assertions = [
        data_quality_assertions_dataset.Assertion(
            assertion=exp.expectation_type,
            success=bool(r.success),
            column=getattr(exp, "column", None),
        )
        for exp, r in zip(expectations, results)
    ]

    end_state = RunState.COMPLETE if all_success else RunState.FAIL
    emit_lineage_event(
        ol_client, run_id, datetime.now(timezone.utc), end_state,
        inputs=[manifest_dataset(df, assertions=assertions)],
    )

    print(f"\nOpenLineage run {run_id} sent to Marquez at {MARQUEZ_URL} "
          f"(namespace={JOB_NAMESPACE}, job={JOB_NAME}) -- check http://localhost:3000")

    if not all_success:
        print("\nOne or more expectations failed -- see above. "
              "The unique md5_hash failure is expected right now (several known "
              "byte-identical duplicate images across the dataset, including one "
              "image carrying two different grade labels); this is the gate doing "
              "its job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
