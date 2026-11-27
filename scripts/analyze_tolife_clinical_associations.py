#!/usr/bin/env python3
"""Describe prior sensor-index associations with actual COPD clinic measures.

This is an exploratory, non-ring statistical analysis. It uses valid daily
sensor indices only from the seven complete days before each clinic visit;
model-estimated CAT/CCQ/mMRC and same-day or future readings are excluded.
Only aggregate counts and participant-level association summaries are saved.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "tolife" / "index.zip"
OUTPUT = ROOT / "models" / "tolife_clinical_associations.json"
SEED = 20261127
LOOKBACK_DAYS = 7
MIN_VALID_DAYS = 3

SENSOR_FEATURES = {
    "BRI.rf_mean": ("BRI.csv", "rf_mean", 4.0, 50.0),
    "BRI.rf_std": ("BRI.csv", "rf_std", 0.0, 25.0),
    "SHRI.hr_mean": ("SHRI.csv", "hr_mean", 25.0, 220.0),
    "PRVI.mean_HR": ("PRVI.csv", "mean_HR", 25.0, 220.0),
    "PRVI.std_RR": ("PRVI.csv", "std_RR", 0.0, 500.0),
    "PRVI.RMSSD": ("PRVI.csv", "RMSSD", 0.0, 500.0),
    "MI.gs_mean": ("MI.csv", "gs_mean", 0.0, 3.0),
    "MI.smwd_mean": ("MI.csv", "smwd_mean", 0.0, 1000.0),
    "MI.walked_time": ("MI.csv", "walked_time", 0.0, 1440.0),
    "MI.walked_dist": ("MI.csv", "walked_dist", 0.0, 100000.0),
}

CLINICAL_TARGETS = {
    "FEV1_pred (%)": "lung function, percent predicted",
    "Lung Ratio": "spirometric FEV1/FVC ratio",
    "Six-Minute Walk Distance (m)": "functional exercise capacity",
    "CAT Score": "clinician-visit COPD symptom score",
    "mMRC Score": "clinician-visit dyspnea grade",
    "CCQ Total Score": "clinician-visit COPD health-status score",
}


def _read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    try:
        with archive.open(name) as stream:
            return pd.read_csv(stream)
    except KeyError:
        return None


def _valid_sensor_series(archive: zipfile.ZipFile, member: str, column: str, lo: float, hi: float) -> pd.DataFrame:
    raw = _read_csv(archive, member)
    if raw is None or not {"Timestamp", column}.issubset(raw.columns):
        return pd.DataFrame(columns=["day", "value"])
    raw["day"] = pd.to_datetime(raw["Timestamp"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    raw["value"] = pd.to_numeric(raw[column], errors="coerce")
    if "index_validity" in raw.columns:
        valid = pd.to_numeric(raw["index_validity"], errors="coerce").eq(1)
        raw = raw[valid]
    raw = raw[raw["value"].between(lo, hi, inclusive="both") & raw["day"].notna()]
    # Multiple sensor rows on one day are reduced without using any clinical labels.
    return raw.groupby("day", as_index=False)["value"].mean()


def build_visit_table(archive_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as archive:
        clinical = _read_csv(archive, "index/clinical_indices_all.csv")
        kits = _read_csv(archive, "index/kits.csv")
        if clinical is None or kits is None:
            raise RuntimeError("TOLIFE index.zip is missing clinical_indices_all.csv or kits.csv")
        clinical["Visit Date"] = pd.to_datetime(clinical["Visit Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        clinical = clinical.dropna(subset=["Kit-ID", "Visit Date"]).copy()
        clinical["Kit-ID"] = clinical["Kit-ID"].astype(str)
        kit_info = kits.set_index(kits["Kit-ID"].astype(str))[["Country"]]
        rows: list[dict[str, Any]] = []
        sensor_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        for _, visit in clinical.iterrows():
            kit = str(visit["Kit-ID"])
            info = kit_info.loc[kit] if kit in kit_info.index else None
            if info is None:
                continue
            country = str(info["Country"]).strip().upper()
            visit_day = visit["Visit Date"]
            out: dict[str, Any] = {
                "participant": kit,
                "visit_day": visit_day,
            }
            for target in CLINICAL_TARGETS:
                out[target] = pd.to_numeric(pd.Series([visit.get(target)]), errors="coerce").iloc[0]
            for feature, (filename, column, lo, hi) in SENSOR_FEATURES.items():
                cache_key = (kit, filename, column)
                if cache_key not in sensor_cache:
                    member = f"index/{country}/{kit}/{filename}"
                    sensor_cache[cache_key] = _valid_sensor_series(archive, member, column, lo, hi)
                series = sensor_cache[cache_key]
                start_day = visit_day - pd.Timedelta(days=LOOKBACK_DAYS)
                prior = series[(series["day"] >= start_day) & (series["day"] < visit_day)]["value"].dropna()
                out[feature] = float(prior.mean()) if len(prior) >= MIN_VALID_DAYS else np.nan
                out[f"{feature}__valid_days"] = int(len(prior))
            rows.append(out)
    visits = pd.DataFrame(rows)
    summary = {
        "clinic_visits_with_sensor_directory": int(len(visits)),
        "participants": int(visits["participant"].nunique()) if len(visits) else 0,
        "lookback_days": LOOKBACK_DAYS,
        "minimum_valid_sensor_days": MIN_VALID_DAYS,
        "same_day_readings_used": False,
        "future_readings_used": False,
        "model_estimated_CAT_CCQ_mMRC_used_as_target_or_input": False,
    }
    return visits, summary


def _participant_correlations(data: pd.DataFrame, feature: str, target: str) -> dict[str, Any]:
    pairs = data[["participant", feature, target]].copy()
    pairs[feature] = pd.to_numeric(pairs[feature], errors="coerce")
    pairs[target] = pd.to_numeric(pairs[target], errors="coerce")
    pairs = pairs.dropna()
    correlations: list[float] = []
    for _, person in pairs.groupby("participant"):
        if len(person) < 3 or person[feature].nunique() < 2 or person[target].nunique() < 2:
            continue
        rho = spearmanr(person[feature], person[target]).statistic
        if np.isfinite(rho):
            correlations.append(float(rho))
    if len(correlations) < 5:
        return {
            "status": "too_few_repeated_participants",
            "matched_visits": int(len(pairs)),
            "participants_with_three_or_more_informative_visits": int(len(correlations)),
        }
    values = np.asarray(correlations, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    bootstrap = np.median(values[rng.integers(0, len(values), size=(4000, len(values)))], axis=1)
    return {
        "status": "descriptive_only",
        "matched_visits": int(len(pairs)),
        "participants_with_three_or_more_informative_visits": int(len(values)),
        "median_within_participant_spearman_rho": float(np.median(values)),
        "participant_bootstrap_95_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "interpretation": "Exploratory association only; no prediction, causality, diagnosis, or ring validation.",
    }


def analyze(visits: pd.DataFrame, counts: dict[str, Any]) -> dict[str, Any]:
    associations: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for target, description in CLINICAL_TARGETS.items():
        target_values = pd.to_numeric(visits[target], errors="coerce")
        coverage[target] = {
            "description": description,
            "visits_with_clinical_measure": int(target_values.notna().sum()),
            "participants_with_clinical_measure": int(visits.loc[target_values.notna(), "participant"].nunique()),
        }
        associations[target] = {}
        for feature in SENSOR_FEATURES:
            associations[target][feature] = _participant_correlations(visits, feature, target)
    return {
        "model_version": "tolife-nonring-clinical-association-0.1",
        "dataset": "TOLIFE Clinical Study A v2.0",
        "dataset_doi": "10.5281/zenodo.23263785",
        "license": "CC BY 4.0",
        "role": "Non-ring clinical association audit only; never used to train or infer a PlumoRing result.",
        "input_device_families": {
            "BRI": "bed/mattress breathing-rate index",
            "SHRI": "sleep heart-rate index",
            "PRVI": "smartwatch PPG-derived pulse-rate variability",
            "MI": "mobility index from the study's heterogeneous devices",
        },
        "excluded_sensor_families": [
            "PEI spirometry measurements, to avoid using lung-function measurements as predictors of lung-function outcomes",
            "Daily model-estimated CAT, CCQ, and mMRC scores",
            "Same-day and future sensor readings",
            "Questionnaire-derived daily estimates and exacerbation labels",
        ],
        "method": "For each clinical visit, average valid sensor indices over the seven complete UTC days before the visit (visit date excluded), requiring at least three valid days. For each sensor-outcome pair, compute a Spearman association within each participant with at least three informative clinical visits; summarize participant correlations by their median and a participant bootstrap interval.",
        "coverage": {**counts, "clinical_targets": coverage},
        "within_participant_associations": associations,
        "limitations": [
            "The cohort has established COPD; these associations do not diagnose COPD or predict exacerbations.",
            "All sensor families are non-ring and heterogeneous; results cannot validate or be transferred directly to PlumoRing.",
            "Participants with fewer than three informative visits do not contribute to within-person correlations.",
            "Bootstrap intervals are descriptive and do not adjust for multiple comparisons.",
            "This does not establish causality or clinical utility.",
        ],
        "reproduction": "python scripts/analyze_tolife_clinical_associations.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"TOLIFE archive not found: {args.archive}")
    print("Reading valid pre-visit sensor indices and measured clinic outcomes...", flush=True)
    visits, counts = build_visit_table(args.archive)
    report = analyze(visits, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["coverage"], indent=2))
    print(f"Wrote aggregate-only association report: {args.output}")


if __name__ == "__main__":
    main()
