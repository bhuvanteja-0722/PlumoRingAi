#!/usr/bin/env python3
"""Evaluate a training-free, patient-personalized pulmonary trend score.

The score uses only prior readings to establish each participant's baseline,
then combines same-day adverse changes in heart rate, breathing rate and SpO2.
It is an exploratory adaptation to sparse TOLIFE daily indices, not the
published BVS3 implementation and not a medical alert or diagnosis.
"""

from __future__ import annotations

import datetime as dt
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_pulmonary_study import _read_csv, _readings


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "tolife" / "index.zip"
REPORT_PATH = ROOT / "models" / "tolife_personalized_trend_metrics.json"
BASELINE_DAYS = 14
MIN_HISTORY = 4
TARGET_DAYS = 7


def _daily_vitals(daily: dict[str, dict[dt.date, dict[str, dict[str, float]]]]) -> dict[str, dict[dt.date, dict[str, float]]]:
    result: dict[str, dict[dt.date, dict[str, float]]] = defaultdict(dict)
    for participant, days in daily.items():
        for day, families in days.items():
            ppg = families.get("ppg", {})
            heart_rate = ppg.get("mean_HR", families.get("heart_rate", {}).get("hr_mean"))
            respiration = families.get("respiration", {}).get("rf_mean")
            oxygen = families.get("oxygen", {}).get("SpO2")
            values = {"heart_rate": heart_rate, "respiration_rate": respiration, "spo2": oxygen}
            clean = {name: float(value) for name, value in values.items() if value is not None and np.isfinite(value)}
            if clean:
                result[participant][day] = clean
    return result


def _robust_z(value: float, history: list[float], scale_floor: float) -> float | None:
    if len(history) < MIN_HISTORY:
        return None
    values = np.asarray(history, dtype=np.float64)
    center = float(np.median(values))
    mad_scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(mad_scale) or mad_scale < scale_floor:
        std_scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        mad_scale = max(std_scale, scale_floor)
    return float(np.clip((value - center) / mad_scale, -5.0, 5.0))


def evaluate(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        daily, _ = _readings(archive)
        events: dict[str, list[dt.date]] = defaultdict(list)
        for event in _read_csv(archive, "index/exacerbations_all.csv"):
            try:
                events[event["Kit-id"]].append(dt.date.fromisoformat(event["Date"]))
            except (KeyError, ValueError):
                continue
        kits = _read_csv(archive, "index/kits.csv")

    vitals = _daily_vitals(daily)
    rows: list[tuple[str, float, int]] = []
    skipped_short_history = 0
    skipped_missing_current = 0
    positive_people: set[str] = set()
    participant_days: dict[str, int] = defaultdict(int)
    positive_days = 0
    scales = {"heart_rate": 3.0, "respiration_rate": 1.5, "spo2": 1.0}
    directions = {"heart_rate": 1.0, "respiration_rate": 1.0, "spo2": -1.0}

    for kit in kits:
        participant = kit.get("Kit-ID", "")
        if participant not in vitals:
            continue
        try:
            first = dt.date.fromisoformat(kit["first_day"])
            last = dt.date.fromisoformat(kit["last_day"])
        except (KeyError, ValueError):
            continue
        dates = sorted(vitals[participant])
        for day in dates:
            if day < first or day > last - dt.timedelta(days=TARGET_DAYS):
                continue
            current = vitals[participant][day]
            z_values: list[float] = []
            enough_any = False
            for channel, direction in directions.items():
                value = current.get(channel)
                if value is None:
                    continue
                history_start = day - dt.timedelta(days=BASELINE_DAYS)
                history = [vitals[participant][prior][channel]
                           for prior in dates
                           if history_start <= prior < day and channel in vitals[participant][prior]]
                if len(history) >= MIN_HISTORY:
                    enough_any = True
                    z = _robust_z(value, history, scales[channel])
                    if z is not None:
                        z_values.append(direction * z)
            if not enough_any or len(z_values) < 2:
                skipped_short_history += 1
                continue
            if not any(name in current for name in directions):
                skipped_missing_current += 1
                continue

            score = float(-norm.logsf(2.0 * float(np.mean(z_values))))
            positive = int(any(dt.timedelta(days=1) <= event_day - day <= dt.timedelta(days=TARGET_DAYS)
                               for event_day in events[participant]))
            rows.append((participant, score, positive))
            participant_days[participant] += 1
            positive_days += positive
            if positive:
                positive_people.add(participant)

    if not rows:
        raise RuntimeError("No TOLIFE days had enough prior history and current readings for a personalized score.")
    groups = np.asarray([row[0] for row in rows])
    scores = np.asarray([row[1] for row in rows], dtype=np.float64)
    labels = np.asarray([row[2] for row in rows], dtype=np.int8)
    group_sizes = {participant: int(np.sum(groups == participant)) for participant in np.unique(groups)}
    weights = np.asarray([1.0 / group_sizes[participant] for participant in groups], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    prevalence = float(np.average(labels, weights=weights))
    metrics = {
        "roc_auc_equal_participant_weight": float(roc_auc_score(labels, scores, sample_weight=weights)) if len(np.unique(labels)) == 2 else None,
        "average_precision_equal_participant_weight": float(average_precision_score(labels, scores, sample_weight=weights)),
        "roc_auc_pooled_days": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "average_precision_pooled_days": float(average_precision_score(labels, scores)),
    }
    return {
        "model_version": "plumoring-personalized-trend-feasibility-0.1",
        "dataset": "TOLIFE Clinical Study A Health Indices v2.0",
        "dataset_doi": "10.5281/zenodo.23263785",
        "task": "research-only ranking of a clinically recorded moderate/severe COPD exacerbation 1–7 days later among people already diagnosed with COPD",
        "method": "Training-free patient-personalized adverse-change composite. For each available vital, current-day values are standardized against that same person's previous 14 calendar days using median and robust MAD scale; an adverse direction is applied (higher HR/RR, lower SpO2), at least two channels and four prior observations per used channel are required, and available channel scores are combined with a monotonic normal-tail transform. No outcome is used to fit or tune the score.",
        "input_channels": ["daily heart-rate index", "daily respiratory-rate index", "daily SpO2 index"],
        "baseline_days": BASELINE_DAYS,
        "target_days_ahead": TARGET_DAYS,
        "counts": {
            "participants": int(len(np.unique(groups))),
            "participants_with_positive_windows": int(len(positive_people)),
            "eligible_participant_days": int(len(rows)),
            "positive_participant_days": int(labels.sum()),
            "positive_prevalence_equal_participant_weight": prevalence,
            "skipped_days_insufficient_prior_channels_or_current_data": int(skipped_short_history),
            "minimum_prior_observations_per_used_channel": MIN_HISTORY,
            "minimum_current_channels": 2,
        },
        "metrics": metrics,
        "no_skill_average_precision_reference_equal_participant_weight": prevalence,
        "deployment_status": "not_deployed_research_only",
        "limitations": [
            "This is an exploratory adaptation of a published personalized wearable-vitals idea to sparse daily TOLIFE indices; it is not an exact reproduction of the published BVS3 pipeline.",
            "TOLIFE did not use the proposed PlumoRing and has no raw ring PPG validation.",
            "This task forecasts exacerbation among people already diagnosed with COPD; it does not diagnose COPD, asthma, pneumonia, infection, or other pulmonary disease.",
            "The data are sparse daily indices rather than the hourly data used in the published 2026 wearable study.",
            "No alert threshold was tuned or enabled. A research ranking score is not clinical advice or a patient risk probability.",
        ],
        "reproduction": "python scripts/evaluate_tolife_personalized_trend.py",
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"Missing TOLIFE archive: {args.archive}")
    report = evaluate(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": report["counts"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
