#!/usr/bin/env python3
"""Evaluate an exploratory COPD-exacerbation task on public TOLIFE data.

This is deliberately an evaluation-only workflow. The available cohort is too
small and the grouped results too weak to export a patient-facing classifier.
Raw data remains in the ignored data/ directory and is never copied to output.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "tolife" / "index.zip"
REPORT_PATH = ROOT / "models" / "pulmonary_feasibility_metrics.json"
TARGET_DAYS = 7
LOOKBACK_DAYS = 7
MIN_MODALITIES = 2

# Keep only channels that have a plausible ring analogue. TOLIFE did not use a
# ring: its channels came from a smartwatch, mattress, shoes/phone, and a
# portable spirometer. FEV1/PEF and all estimated CAT/CCQ/mMRC scores are
# intentionally excluded to avoid lung-function tests and derived-label leak.
SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "BRI.csv": ("respiration", ("rf_mean", "rf_std")),
    "PRVI.csv": ("ppg", ("mean_HR", "std_RR", "RMSSD", "nRMSSD")),
    "MI.csv": ("movement", ("gs_mean", "sl_mean", "smwd_mean", "walked_time", "walked_dist")),
    "PEI.csv": ("oxygen", ("SpO2",)),
    "SHRI.csv": ("heart_rate", ("hr_mean",)),
}
FEATURES = tuple(dict.fromkeys(column for _, columns in SOURCES.values() for column in columns))


def _read_csv(archive: zipfile.ZipFile, path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(path).decode("utf-8-sig"))))


def _readings(archive: zipfile.ZipFile) -> tuple[dict[str, dict[dt.date, dict[str, dict[str, float]]]], set[str]]:
    daily: dict[str, dict[dt.date, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    participants: set[str] = set()
    for path in archive.namelist():
        filename = path.rsplit("/", 1)[-1]
        if filename not in SOURCES or path.count("/") < 3:
            continue
        kit = path.split("/")[-2]
        participants.add(kit)
        family, columns = SOURCES[filename]
        for row in _read_csv(archive, path):
            # index_validity is the dataset's quality flag. data_validity alone
            # can mean a row exists even when its derived index is unusable.
            if row.get("index_validity") != "1":
                continue
            try:
                day = dt.date.fromisoformat(row["Timestamp"][:10])
            except (KeyError, ValueError):
                continue
            values: dict[str, float] = {}
            for name in columns:
                try:
                    value = float(row[name])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                if name in {"mean_HR", "hr_mean", "SpO2", "rf_mean"} and value <= 0:
                    continue
                values[name] = value
            if values:
                daily[kit][day][family] = values
    return daily, participants


def _build_examples(archive: zipfile.ZipFile):
    kits = _read_csv(archive, "index/kits.csv")
    events = _read_csv(archive, "index/exacerbations_all.csv")
    daily, participant_dirs = _readings(archive)
    event_dates: dict[str, list[dt.date]] = defaultdict(list)
    event_types: dict[str, int] = defaultdict(int)
    for event in events:
        kit = event.get("Kit-id", "")
        try:
            event_dates[kit].append(dt.date.fromisoformat(event["Date"]))
            event_types[event.get("Type", "Unknown")] += 1
        except (KeyError, ValueError):
            continue

    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    example_days: list[str] = []
    modality_counts: dict[str, int] = defaultdict(int)
    eligible_participants: set[str] = set()
    positive_participants: set[str] = set()
    for kit_row in kits:
        kit = kit_row["Kit-ID"]
        if kit not in daily:
            continue
        try:
            start = dt.date.fromisoformat(kit_row["first_day"])
            end = dt.date.fromisoformat(kit_row["last_day"])
        except (KeyError, ValueError):
            continue
        dates = sorted(daily[kit])
        for day in dates:
            # Predict only from measurements at/before the index day; the
            # target excludes same-day events and looks forward 1..7 days.
            if day < start or day > end - dt.timedelta(days=TARGET_DAYS):
                continue
            history = [daily[kit].get(day - dt.timedelta(days=offset), {}) for offset in range(LOOKBACK_DAYS - 1, -1, -1)]
            modalities = {source for reading in history for source in reading}
            if len(modalities) < MIN_MODALITIES:
                continue
            features: list[float] = []
            for name in FEATURES:
                values = [reading[source][name] for reading in history for source in reading if name in reading[source]]
                features.extend((float(np.mean(values)) if values else np.nan, float(np.std(values)) if values else np.nan))
            future_event = any(dt.timedelta(days=1) <= event_day - day <= dt.timedelta(days=TARGET_DAYS) for event_day in event_dates[kit])
            rows.append(features)
            labels.append(int(future_event))
            groups.append(kit)
            example_days.append(day.isoformat())
            eligible_participants.add(kit)
            modality_counts[str(len(modalities))] += 1
            if future_event:
                positive_participants.add(kit)

    if not rows:
        raise RuntimeError("No eligible participant-days found. Confirm that data/tolife/index.zip is the TOLIFE v2.0 index archive.")
    return {
        "X": np.asarray(rows, dtype=np.float64),
        "y": np.asarray(labels, dtype=np.int8),
        "groups": np.asarray(groups),
        "example_days": example_days,
        "participants": participant_dirs,
        "eligible_participants": eligible_participants,
        "positive_participants": positive_participants,
        "event_types": event_types,
        "event_rows": len(events),
        "modality_counts": dict(sorted(modality_counts.items())),
    }


def _models() -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=2026),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=5, random_state=2026),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.7, class_weight="balanced_subsample", n_jobs=-1, random_state=2026),
        ),
    }


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    X, y, groups = data["X"], data["y"], data["groups"]
    if len(set(groups)) < 5 or len(set(y)) < 2:
        raise RuntimeError("At least five participants and both outcome classes are required for grouped evaluation.")
    splitter = GroupKFold(n_splits=5)
    results: dict[str, Any] = {}
    for name, model in _models().items():
        oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_results = []
        for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
            if len(np.unique(y[train])) < 2:
                continue
            fit_options = {}
            if name == "hist_gradient_boosting":
                fit_options["histgradientboostingclassifier__sample_weight"] = compute_sample_weight("balanced", y[train])
            model.fit(X[train], y[train], **fit_options)
            predictions = model.predict_proba(X[test])[:, 1]
            oof[test] = predictions
            fold_results.append({
                "fold": fold,
                "held_out_participants": int(len(set(groups[test]))),
                "positive_days": int(y[test].sum()),
                "days": int(len(test)),
                "roc_auc": float(roc_auc_score(y[test], predictions)) if len(np.unique(y[test])) == 2 else None,
                "average_precision": float(average_precision_score(y[test], predictions)),
            })
        valid = np.isfinite(oof)
        valid_labels = y[valid]
        valid_scores = oof[valid]
        results[name] = {
            "grouped_oof_roc_auc": float(roc_auc_score(valid_labels, valid_scores)) if len(np.unique(valid_labels)) == 2 else None,
            "grouped_oof_average_precision": float(average_precision_score(valid_labels, valid_scores)),
            "folds": fold_results,
        }

    prevalence = float(y.mean())
    # A candidate is considered only an experiment. We intentionally do not
    # save weights or permit scoring when none clearly exceeds the naive AP.
    best_ap = max(result["grouped_oof_average_precision"] for result in results.values())
    best_auc = max((result["grouped_oof_roc_auc"] or 0) for result in results.values())
    ready = best_ap > prevalence * 1.25 and best_auc >= 0.60
    return {
        "model_version": "plumoring-copd-exacerbation-feasibility-0.1",
        "dataset": "TOLIFE Clinical Study A Health Indices v2.0",
        "dataset_doi": "10.5281/zenodo.23263785",
        "task": "research-only prediction of a clinically recorded moderate/severe COPD exacerbation 1–7 days after an observation day, among participants already diagnosed with COPD",
        "input_channels": list(FEATURES),
        "lookback_days": LOOKBACK_DAYS,
        "target_days_ahead": TARGET_DAYS,
        "minimum_valid_sensor_families": MIN_MODALITIES,
        "split": "5-fold GroupKFold by participant kit; all rows from a participant stay in one fold",
        "counts": {
            "participants_with_reading_files": len(data["participants"]),
            "participants_with_eligible_days": len(data["eligible_participants"]),
            "participants_with_positive_days": len(data["positive_participants"]),
            "eligible_participant_days": int(len(y)),
            "positive_participant_days": int(y.sum()),
            "positive_prevalence": prevalence,
            "recorded_exacerbation_events": int(data["event_rows"]),
            "events_by_severity": dict(data["event_types"]),
            "days_by_valid_sensor_family_count": data["modality_counts"],
        },
        "models": results,
        "naive_average_precision_reference": prevalence,
        "candidate_meets_exploratory_gate": bool(ready),
        "deployment_model_exported": False,
        "deployment_status": "not_ready",
        "reason": "Grouped participant-held-out results do not exceed the no-skill event-prevalence reference by the prespecified margin and AUC gate. No patient-facing risk score is exported.",
        "limitations": [
            "This is a COPD exacerbation forecasting feasibility task among people already known to have COPD; it does not diagnose COPD or identify asthma, pneumonia, infection, or other pulmonary diseases.",
            "TOLIFE uses smartwatch, mattress, mobility, and spirometer-related sensors, not the proposed finger ring. Results do not validate ring transfer.",
            "Only HR/HRV, respiratory-rate, movement, and SpO2-like channels are used. FEV1, PEF, estimated CAT/CCQ/mMRC, audio, environmental, demographic, and clinical-visit fields are excluded.",
            "Daily indices are sparse and the eligible event-positive examples come from few participants. There is no external validation cohort in this evaluation.",
            "Scores are experimental ranking outputs, not calibrated probabilities, medical alerts, diagnosis, or care advice.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="Path to the TOLIFE v2.0 index.zip file")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Path for aggregate metrics JSON")
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"Dataset not found: {args.archive}\nDownload the TOLIFE v2.0 index.zip listed in README.md.")
    with zipfile.ZipFile(args.archive) as archive:
        report = evaluate(_build_examples(archive))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote aggregate metrics only: {args.output}")


if __name__ == "__main__":
    main()
