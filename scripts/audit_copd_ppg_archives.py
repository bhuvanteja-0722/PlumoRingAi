#!/usr/bin/env python3
"""Summarize subject-file coverage in the downloaded public COPD PPG archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVES = tuple(
    ROOT / f"data/copd_ppg_zenodo/{duration}s_db.zip"
    for duration in (2, 16, 32, 64, 128, 256)
)


def inspect(path: Path) -> dict[str, object]:
    subjects = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            content = io.StringIO(archive.read(member).decode("utf-8-sig"))
            reader = csv.reader(content, delimiter=";")
            header = next(reader, None)
            labels: dict[str, int] = {}
            row_count = 0
            for row in reader:
                if not row:
                    continue
                row_count += 1
                label = row[-1]
                labels[label] = labels.get(label, 0) + 1
            expected = "1" if Path(member).stem.startswith("P_") else "0"
            subjects.append({
                "file": Path(member).name,
                "rows": row_count,
                "feature_count": max((len(header) if header else 0) - 1, 0),
                "expected_label": expected,
                "label_counts": labels,
                "labels_match_filename": set(labels) == {expected},
            })
    return {
        "archive": path.name,
        "subject_file_count": len(subjects),
        "independent_people": len(subjects),
        "files": subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path, default=DEFAULT_ARCHIVES)
    args = parser.parse_args()
    missing = [str(path) for path in args.archives if not path.is_file()]
    if missing:
        raise SystemExit("Download all six duration archives from the official Zenodo COPD PPG record. Missing: " + ", ".join(missing))
    print(json.dumps([inspect(path) for path in args.archives], indent=2))


if __name__ == "__main__":
    main()
