"""
Phase 3 exit-gate check: runs a Great Expectations validation suite
against the manifest built by build_manifest.py, then emits an
OpenLineage START/COMPLETE run event to Marquez so the run shows up
in Marquez's lineage graph.

Usage:
  python build_manifest.py       # produces manifest.csv first
  python ge_validate.py

This is deliberately checking for the real duplicate-image issue found
earlier in the raw dataset (Day 3 Grade B, GBR_202/GBR_203 byte-identical)
-- so the "unique md5_hash" expectation is EXPECTED to fail on the
current dataset. That failure is the point: this is GE catching, in an
automated gate, a data-quality bug we previously only found by hand.
"""
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
import great_expectations as gx

from openlineage.client import OpenLineageClient
from openlineage.client.event_v2 import RunEvent, RunState, Run, Job
from openlineage.client.uuid import generate_new_uuid

MARQUEZ_URL = "http://localhost:5000"
JOB_NAMESPACE = "garbhaai"
JOB_NAME = "data_quality.validate_raw_manifest"


def emit_lineage_event(ol_client, run_id, event_time, state, extra=None):
    ol_client.emit(
        RunEvent(
            eventType=state,
            eventTime=event_time.isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=JOB_NAMESPACE, name=JOB_NAME),
            producer="https://github.com/Hrishi0929/garbhaai-platform/training/data_quality",
        )
    )


def main():
    df = pd.read_csv("manifest.csv")

    ol_client = OpenLineageClient(url=MARQUEZ_URL)
    run_id = str(generate_new_uuid())
    emit_lineage_event(ol_client, run_id, datetime.now(timezone.utc), RunState.START)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("manifest_source")
    data_asset = data_source.add_dataframe_asset("manifest")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_manifest")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="image_id"),
        gx.expectations.ExpectColumnValuesToBeInSet(column="grade", value_set=["A", "B", "C"]),
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

    end_state = RunState.COMPLETE if all_success else RunState.FAIL
    emit_lineage_event(ol_client, run_id, datetime.now(timezone.utc), end_state)

    print(f"\nOpenLineage run {run_id} sent to Marquez at {MARQUEZ_URL} "
          f"(namespace={JOB_NAMESPACE}, job={JOB_NAME}) -- check http://localhost:3000")

    if not all_success:
        print("\nOne or more expectations failed -- see above. "
              "The unique md5_hash failure is expected right now (known duplicate "
              "image in Day 3 Grade B); this is the gate doing its job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
