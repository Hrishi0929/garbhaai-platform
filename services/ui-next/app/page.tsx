"use client";

import { useEffect, useState } from "react";
import type { GradeResult, IngestResult, QualityResult } from "@/lib/types";

// Day 5 is left out on purpose: the model was trained on the pooled Day 3 + Day 4
// data only, and the inference service rejects day=5.
const DAYS = [3, 4];
const GRADES = ["A", "B", "C"];

async function post<T>(path: string, body: FormData): Promise<T> {
  const res = await fetch(path, { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(typeof err.detail === "string" ? err.detail : `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export default function Home() {
  const [patientId, setPatientId] = useState("");
  const [day, setDay] = useState(DAYS[0]);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [quality, setQuality] = useState<QualityResult | null>(null);
  const [ingest, setIngest] = useState<IngestResult | null>(null);
  const [grade, setGrade] = useState<GradeResult | null>(null);
  const [gradeDay, setGradeDay] = useState(DAYS[0]);
  const [gradePatient, setGradePatient] = useState("");
  const [override, setOverride] = useState(GRADES[0]);
  const [saved, setSaved] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return setPreviewUrl(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function reset() {
    setQuality(null);
    setIngest(null);
    setGrade(null);
    setSaved(null);
    setError(null);
  }

  async function onFile(f: File | null) {
    setFile(f);
    reset();
    if (!f) return;
    setBusy("Checking image quality...");
    try {
      const form = new FormData();
      form.append("file", f);
      setQuality(await post<QualityResult>("/api/quality-check", form));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function process() {
    if (!file) return;
    setError(null);
    setSaved(null);
    try {
      setBusy("Ingesting...");
      const ingestForm = new FormData();
      ingestForm.append("file", file);
      ingestForm.append("patient_id", patientId);
      ingestForm.append("day", String(day));
      setIngest(await post<IngestResult>("/api/ingest", ingestForm));

      setBusy("Grading...");
      const gradeForm = new FormData();
      gradeForm.append("file", file);
      gradeForm.append("day", String(day));
      setGrade(await post<GradeResult>("/api/grade", gradeForm));
      setGradeDay(day);
      setGradePatient(patientId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function review(finalGrade: string, overridden: boolean) {
    if (!grade) return;
    setError(null);
    try {
      setBusy("Saving...");
      const form = new FormData();
      form.append("image_hash", grade.image_hash);
      form.append("day", String(gradeDay));
      form.append("model_grade", grade.grade);
      form.append("model_confidence", String(grade.confidence));
      form.append("model_probabilities", JSON.stringify(grade.probabilities));
      form.append("final_grade", finalGrade);
      form.append("doctor_overridden", String(overridden));
      form.append("patient_id", gradePatient);
      await post("/api/review", form);
      setSaved(
        overridden
          ? `Saved. Corrected to ${finalGrade}. Future uploads of this image will use this grade.`
          : "Saved. This image's grade is now confirmed.",
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const canProcess = !!file && !!quality?.ok && patientId.trim() !== "" && !busy;

  return (
    <main>
      <h1>GarbhaAI embryo grading</h1>
      <p className="sub">Upload an embryo image to run it through quality check, ingestion and grading.</p>

      <section className="card">
        <div className="field">
          <label htmlFor="patient">Patient ID</label>
          <input id="patient" type="text" value={patientId} onChange={(e) => setPatientId(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="day">Day</label>
          <select id="day" value={day} onChange={(e) => setDay(Number(e.target.value))}>
            {DAYS.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="file">Upload embryo image</label>
          <input id="file" type="file" accept=".jpg,.jpeg,.png" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
          {previewUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img className="preview" src={previewUrl} alt="Uploaded embryo" />
          )}
        </div>

        {quality && !quality.ok && (
          <div className="notice err">{quality.warnings[0] ?? "Image failed the quality check."}</div>
        )}
        {quality?.ok && quality.warnings.map((w) => (
          <div className="notice warn" key={w}>{w}</div>
        ))}

        <div className="row" style={{ marginTop: 16 }}>
          <button onClick={process} disabled={!canProcess}>Process image</button>
          {busy && <span className="sub" style={{ margin: 0 }}>{busy}</span>}
        </div>
        {error && <div className="notice err">{error}</div>}
        {ingest && (
          <div className={`notice ${ingest.already_seen ? "info" : "ok"}`}>
            {ingest.already_seen
              ? `This exact image was already ingested previously (hash ${ingest.image_hash}).`
              : `Stored new image, hash ${ingest.image_hash}.`}
          </div>
        )}
      </section>

      {grade && (
        <section className="card">
          <p className="confidence" style={{ margin: 0 }}>Model grade</p>
          <p className="grade">{grade.grade}</p>
          <p className="confidence">
            {Math.round(grade.confidence * 100)}% confident that the grade is {grade.grade}
          </p>
          <div className="bars" aria-label="Class probabilities">
            {Object.entries(grade.probabilities).map(([k, v]) => (
              <div className="bar-row" key={k}>
                <span>{k}</span>
                <div className="bar-track"><div className="bar-fill" style={{ width: `${v * 100}%` }} /></div>
                <span>{(v * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>

          {grade.source === "cached_review" ? (
            <div className="notice info">
              This exact image was already reviewed previously
              {grade.doctor_overridden ? " (doctor-overridden)" : ""}. Showing that confirmed grade.
            </div>
          ) : (
            <>
              <label>Confirm or correct this grade</label>
              <div className="row">
                <button onClick={() => review(grade.grade, false)} disabled={!!busy}>Accept</button>
                <select aria-label="Correct to" value={override} onChange={(e) => setOverride(e.target.value)} style={{ width: "auto" }}>
                  {GRADES.map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
                <button className="secondary" onClick={() => review(override, true)} disabled={!!busy}>
                  Submit correction
                </button>
              </div>
            </>
          )}
          {saved && <div className="notice ok">{saved}</div>}
        </section>
      )}
    </main>
  );
}
