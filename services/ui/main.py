"""
UI service: thin Streamlit client over the Phase 4/5 microservices.

Ported from garbhaai-embryo-grading/src/app.py, but all business logic
(quality check, preprocessing, feature extraction, classification, the
image-hash re-grading cache, the doctor review record) has moved out
into separate HTTP services -- this app's job is the upload form,
calling those services in sequence, and rendering what they return. It
computes nothing itself.

Day 5 is left out of the day selector on purpose: the inference
service's model was only ever trained on the pooled Day3+Day4 dataset,
and /grade rejects day=5 with a 422 -- there's no working grade to show
for it yet.
"""
import json
import os

import requests
import streamlit as st

INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8001")
QUALITY_CHECK_URL = os.environ.get("QUALITY_CHECK_URL", "http://localhost:8002")
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8005")

st.set_page_config(page_title="GarbhaAI", layout="centered")


def call_quality_check(image_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        f"{QUALITY_CHECK_URL}/check-quality",
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_ingest(image_bytes: bytes, filename: str, patient_id: str, day: int) -> dict:
    resp = requests.post(
        f"{INGESTION_URL}/ingest",
        data={"patient_id": patient_id, "day": day},
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_grade(image_bytes: bytes, filename: str, day: int) -> dict:
    resp = requests.post(
        f"{INFERENCE_URL}/grade",
        data={"day": day},
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_review(review_payload: dict) -> dict:
    resp = requests.post(f"{INFERENCE_URL}/review", data=review_payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_review_payload(
    grade_result: dict, day: int, patient_id: str, final_grade: str, doctor_overridden: bool
) -> dict:
    """Pure and testable without a running server -- the HTTP call
    itself (call_review) is just this payload plus a POST."""
    return {
        "image_hash": grade_result["image_hash"],
        "day": day,
        "model_grade": grade_result["grade"],
        "model_confidence": grade_result["confidence"],
        "model_probabilities": json.dumps(grade_result["probabilities"]),
        "final_grade": final_grade,
        "doctor_overridden": doctor_overridden,
        "patient_id": patient_id,
    }


def render_grade_result(grade_result: dict, day: int, patient_id: str):
    st.metric("Model grade", grade_result["grade"])
    confidence_pct = grade_result["confidence"]
    st.caption(f"{confidence_pct:.0%} confident that the grade is {grade_result['grade']}")
    st.bar_chart(grade_result["probabilities"])

    if grade_result["source"] == "cached_review":
        note = " (doctor-overridden)" if grade_result.get("doctor_overridden") else ""
        st.info(
            f"This exact image was already reviewed previously{note} -- "
            "showing that confirmed grade."
        )
        return

    st.write("Confirm or correct this grade:")
    col_accept, col_override = st.columns([1, 2])
    if col_accept.button("Accept", key=f"accept-{grade_result['image_hash']}"):
        payload = build_review_payload(grade_result, day, patient_id, grade_result["grade"], False)
        call_review(payload)
        st.success("Saved -- this image's grade is now confirmed.")

    override_key = f"override-select-{grade_result['image_hash']}"
    override_grade = col_override.selectbox("Correct to", options=["A", "B", "C"], key=override_key)
    if col_override.button("Submit correction", key=f"override-{grade_result['image_hash']}"):
        payload = build_review_payload(grade_result, day, patient_id, override_grade, True)
        call_review(payload)
        st.success(
            f"Saved -- corrected to {override_grade}. "
            "Future uploads of this image will use this grade."
        )


def main():
    st.title("GarbhaAI -- embryo grading")
    st.caption("Upload an embryo image to run it through quality-check, ingestion, and grading.")

    patient_id = st.text_input("Patient ID")
    day = st.selectbox("Day", options=[3, 4])
    uploaded_file = st.file_uploader("Upload embryo image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Uploaded image", width=300)

    quality = call_quality_check(image_bytes, uploaded_file.name)
    if not quality["ok"]:
        st.error(
            quality["warnings"][0] if quality["warnings"] else "Image failed the quality check."
        )
        return
    for warning in quality["warnings"]:
        st.warning(warning)

    can_process = bool(patient_id.strip()) and quality["ok"]
    if st.button("Process image", disabled=not can_process):
        with st.spinner("Ingesting..."):
            ingest_result = call_ingest(image_bytes, uploaded_file.name, patient_id, day)
        if ingest_result["already_seen"]:
            st.info(
                "This exact image was already ingested previously "
                f"(hash {ingest_result['image_hash']})."
            )
        else:
            st.success(f"Stored new image, hash {ingest_result['image_hash']}.")

        with st.spinner("Grading..."):
            st.session_state["grade_result"] = call_grade(image_bytes, uploaded_file.name, day)
        st.session_state["grade_day"] = day
        st.session_state["grade_patient_id"] = patient_id

    grade_result = st.session_state.get("grade_result")
    if grade_result is not None:
        render_grade_result(
            grade_result,
            st.session_state.get("grade_day", day),
            st.session_state.get("grade_patient_id", patient_id),
        )


if __name__ == "__main__":
    main()
