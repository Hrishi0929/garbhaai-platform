"""
UI service: thin Streamlit client over the Phase 4 microservices.

Ported from garbhaai-embryo-grading/src/app.py, but the business logic
(quality check, preprocessing, feature extraction) has all moved out
into separate HTTP services -- this app's job is now just the upload
form, calling those services, and rendering what they return. It does
NOT compute anything itself.

Deliberately stops at "features extracted" rather than showing a final
grade: producing grades is the Phase 5 "inference" service's job (the
classifier head, doctor-review flow, and the image-hash re-grading
cache all live there, not here). Wiring this screen up to a real grade
is the first thing Phase 5 adds on top of this file.
"""
import os

import requests
import streamlit as st

INGESTION_URL = os.environ.get("INGESTION_URL", "http://localhost:8001")
QUALITY_CHECK_URL = os.environ.get("QUALITY_CHECK_URL", "http://localhost:8002")
FEATURE_EXTRACTION_URL = os.environ.get("FEATURE_EXTRACTION_URL", "http://localhost:8004")

st.set_page_config(page_title="GarbhaAI (Phase 4 preview)", layout="centered")


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


def call_extract_features(image_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        f"{FEATURE_EXTRACTION_URL}/extract-features",
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def summarize_features(features: list) -> str:
    """A short, human-readable preview of a 512-dim vector -- showing
    all 512 numbers would be useless to a person looking at this screen."""
    preview = ", ".join(f"{v:.3f}" for v in features[:5])
    return f"[{preview}, ... ({len(features)} dims total)]"


def main():
    st.title("GarbhaAI -- embryo image pipeline (Phase 4 preview)")
    st.caption(
        "This screen exercises ingestion -> quality-check -> feature-extraction "
        "as separate services. Final grading moves here in Phase 5."
    )

    patient_id = st.text_input("Patient ID")
    day = st.selectbox("Day", options=[3, 4, 5])
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

    can_process = bool(patient_id.strip()) and (quality["ok"])
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

        with st.spinner("Extracting features..."):
            feature_result = call_extract_features(image_bytes, uploaded_file.name)
        st.metric("Feature vector dimension", feature_result["feature_dim"])
        st.code(summarize_features(feature_result["features"]))
        st.caption("Grading against these features arrives with the Phase 5 inference service.")


if __name__ == "__main__":
    main()
