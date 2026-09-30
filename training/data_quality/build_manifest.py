"""
Phase 3: builds a flat manifest of the raw embryo image dataset so
Great Expectations has a tabular dataset to validate (image quality
checks, class balance, duplicate detection) instead of a folder of
images GE can't reason about directly.

This walks the Day3/Day4/Day5 dataset folders (same layout as
../../../Dataset -- Day 3/4/5 Dataset > Grade A/B/C > images), computes
an MD5 hash per file, and writes one row per image to a CSV.

Usage:
  python build_manifest.py                          # uses default path below
  python build_manifest.py --raw-root /some/other/path
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[3] / "Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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

    df = build_manifest(args.raw_root)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(df["grade"].value_counts())
    print(f"duplicate md5 hashes: {df['md5_hash'].duplicated().sum()}")


if __name__ == "__main__":
    main()
