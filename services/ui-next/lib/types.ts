export type QualityResult = { ok: boolean; warnings: string[] };

export type IngestResult = {
  image_id: string;
  image_hash: string;
  storage_key: string;
  already_seen: boolean;
};

export type GradeResult = {
  grade: string;
  confidence: number;
  probabilities: Record<string, number>;
  image_hash: string;
  source: string;
  doctor_overridden?: boolean;
};
