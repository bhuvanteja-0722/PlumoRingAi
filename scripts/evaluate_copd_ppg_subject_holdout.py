#!/usr/bin/env python3
"""Run a research-only, person-held-out COPD/PPG feasibility comparison.

Raw public data stay in the ignored data/ directory. The script samples a fixed,
deterministic number of windows per person, evaluates six duration archives
separately, and writes aggregate person-level metrics only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "copd_ppg_zenodo"
OUTPUT_PATH = ROOT / "models" / "copd_ppg_screening_metrics.json"
DURATIONS = (2, 16, 32, 64, 128, 256)
SEED = 20261127
DEFAULT_WINDOW_LIMIT = 512


def sample_windows(archive_path: Path, limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read each participant file and keep a reproducible uniform window sample."""
    people: list[dict[str, Any]] = []
    archive_summary: dict[str, Any] = {"archive": archive_path.name, "people": 0, "rows": 0}
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        for member in members:
            stem = Path(member).stem
            if stem.startswith("P"):
                expected_label = 1
            elif stem.startswith("N"):
                expected_label = 0
            else:
                raise ValueError(f"Unexpected participant file name in {archive_path.name}: {member}")

            local_seed = zlib.crc32(f"{seed}:{archive_path.name}:{member}".encode("utf-8"))
            rng = random.Random(local_seed)
            selected: list[list[float]] = []
            observed_rows = 0
            label_values: set[str] = set()
            with archive.open(member) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"), delimiter=";")
                header = next(reader, None)
                if not header or len(header) < 3:
                    raise ValueError(f"Missing PPG feature columns in {archive_path.name}:{member}")
                feature_count = len(header) - 1
                for row in reader:
                    if not row:
                        continue
                    if len(row) != len(header):
                        raise ValueError(f"Inconsistent row width in {archive_path.name}:{member}")
                    label = row[-1].strip()
                    label_values.add(label)
                    if label not in {"0", "1"} or int(label) != expected_label:
                        raise ValueError(f"Filename/label mismatch in {archive_path.name}:{member}")
                    try:
                        window = [float(value) for value in row[:-1]]
                    except ValueError as exc:
                        raise ValueError(f"Non-numeric PPG sample in {archive_path.name}:{member}") from exc
                    if not np.isfinite(window).all():
                        continue
                    observed_rows += 1
                    if len(selected) < limit:
                        selected.append(window)
                    else:
                        replace_at = rng.randrange(observed_rows)
                        if replace_at < limit:
                            selected[replace_at] = window

            if label_values != {str(expected_label)}:
                raise ValueError(f"Expected one verified class label in {archive_path.name}:{member}")
            if not selected:
                raise ValueError(f"No valid PPG windows found in {archive_path.name}:{member}")
            raw_windows = np.asarray(selected, dtype=np.float64)
            people.append({
                "group": stem,
                "label": expected_label,
                "rows": observed_rows,
                "sampled_windows": len(selected),
                "feature_count": feature_count,
                "features": waveform_features(raw_windows),
            })
            archive_summary["rows"] += observed_rows

    archive_summary["people"] = len(people)
    archive_summary["copd_people"] = sum(person["label"] == 1 for person in people)
    archive_summary["control_people"] = sum(person["label"] == 0 for person in people)
    if archive_summary["people"] != 6 or archive_summary["copd_people"] != 4 or archive_summary["control_people"] != 2:
        raise ValueError(
            f"Expected the released 4 COPD + 2 control subject files in {archive_path.name}; "
            f"found {archive_summary['people']} people, {archive_summary['copd_people']} COPD, "
            f"{archive_summary['control_people']} controls."
        )
    return people, archive_summary


def waveform_features(windows: np.ndarray) -> np.ndarray:
    """Extract scale-normalized morphology and normalized spectral summaries."""
    mean = windows.mean(axis=1, keepdims=True)
    centered = windows - mean
    scale = np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
    normalized = centered / scale
    first = np.diff(normalized, axis=1)
    second = np.diff(first, axis=1)

    quantiles = np.quantile(normalized, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    skew = np.mean(normalized**3, axis=1)
    excess_kurtosis = np.mean(normalized**4, axis=1) - 3.0
    crossings = np.mean((normalized[:, 1:] * normalized[:, :-1]) < 0, axis=1)
    peaks = np.mean(
        (normalized[:, 1:-1] > normalized[:, :-2])
        & (normalized[:, 1:-1] >= normalized[:, 2:]),
        axis=1,
    )
    spectral = np.abs(np.fft.rfft(normalized, axis=1)) ** 2
    spectral[:, 0] = 0.0
    total_power = np.maximum(spectral.sum(axis=1, keepdims=True), 1e-12)
    normalized_power = spectral / total_power
    band_powers = np.stack(
        [normalized_power[:, a:b].sum(axis=1) for a, b in ((1, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 129))],
        axis=1,
    )
    spectral_entropy = -np.sum(
        normalized_power * np.log(np.maximum(normalized_power, 1e-12)), axis=1
    ) / np.log(normalized_power.shape[1] - 1)

    summary = np.column_stack((
        quantiles,
        skew,
        excess_kurtosis,
        np.mean(np.abs(first), axis=1),
        np.std(first, axis=1),
        np.sqrt(np.mean(first**2, axis=1)),
        np.mean(np.abs(second), axis=1),
        np.std(second, axis=1),
        crossings,
        peaks,
        np.max(np.abs(normalized), axis=1),
        spectral_entropy,
        band_powers,
    ))
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


def build_models() -> dict[str, Any]:
    return {
        "regularized_logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(10, 90)),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=400, random_state=SEED),
        ),
        "rbf_support_vector": make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(10, 90)),
            SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=120,
                max_features="sqrt",
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
    }


def decision_scores(model: Any, features: np.ndarray) -> tuple[np.ndarray, float]:
    classifier = model.steps[-1][1]
    if hasattr(classifier, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float), 0.0
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float), 0.5


def evaluate(people: list[dict[str, Any]]) -> dict[str, Any]:
    participant_labels = np.asarray([person["label"] for person in people], dtype=int)
    results: dict[str, Any] = {}
    for model_name in build_models():
        oof_scores = np.full(len(people), np.nan, dtype=float)
        thresholds: list[float] = []
        for held_out, person in enumerate(people):
            train_people = [item for index, item in enumerate(people) if index != held_out]
            train_x = np.vstack([item["features"] for item in train_people])
            train_y = np.concatenate([
                np.full(len(item["features"]), item["label"], dtype=int) for item in train_people
            ])
            test_x = person["features"]
            model = build_models()[model_name]
            model.fit(train_x, train_y)
            window_scores, threshold = decision_scores(model, test_x)
            oof_scores[held_out] = float(np.median(window_scores))
            thresholds.append(threshold)

        predicted = np.asarray([
            score >= threshold for score, threshold in zip(oof_scores, thresholds)
        ], dtype=int)
        tn, fp, fn, tp = confusion_matrix(participant_labels, predicted, labels=[0, 1]).ravel()
        results[model_name] = {
            "validation": "leave-one-participant-out",
            "independent_people_tested": int(len(participant_labels)),
            "participant_level_roc_auc": round(float(roc_auc_score(participant_labels, oof_scores)), 4),
            "participant_level_average_precision": round(float(average_precision_score(participant_labels, oof_scores)), 4),
            "participant_level_balanced_accuracy": round(float(balanced_accuracy_score(participant_labels, predicted)), 4),
            "participant_level_accuracy_at_default_threshold": round(float(accuracy_score(participant_labels, predicted)), 4),
            "sensitivity_at_default_threshold": round(float(tp / (tp + fn)) if tp + fn else 0.0, 4),
            "specificity_at_default_threshold": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
            "confusion_counts": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
            "score_note": "Median held-out window score per participant; scores are not calibrated probabilities.",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-limit", type=int, default=DEFAULT_WINDOW_LIMIT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.window_limit < 100:
        parser.error("--window-limit must be at least 100 windows per participant")

    archive_reports: dict[str, Any] = {}
    for duration in DURATIONS:
        archive_path = DATA_DIR / f"{duration}s_db.zip"
        if not archive_path.is_file():
            raise SystemExit(
                f"Missing {archive_path}. Download the public duration archives from "
                "https://zenodo.org/records/18109876 and place them in data/copd_ppg_zenodo/."
            )
        people, summary = sample_windows(archive_path, args.window_limit, SEED)
        archive_reports[f"{duration}s"] = {
            **summary,
            "sampled_windows_per_person_max": args.window_limit,
            "models": evaluate(people),
        }

    report = {
        "model_version": "plumoring-copd-ppg-lopo-v0.1",
        "dataset": "Ultra-Short PPG Signal Dataset for COPD Screening, Zenodo 18109876",
        "source_url": "https://zenodo.org/records/18109876",
        "task": "Research-only COPD-versus-control classification from PPG segments; cross-sectional diagnosis feasibility, not exacerbation forecasting.",
        "label_mapping": "Verified against each CSV: P_* files are COPD (1); N_* files are controls (0).",
        "independent_participants_per_archive": 6,
        "cohort_per_archive": {"copd": 4, "controls": 2},
        "window_feature_method": "Scale-normalized waveform shape, derivative, peak, and normalized spectrum summaries; no demographics or auxiliary labels.",
        "validation": "Leave-one-participant-out; every window from the held-out person stays out of training. The participant score is the median of sampled held-out window scores. Each duration archive is evaluated separately.",
        "window_sampling": {
            "strategy": "Deterministic reservoir sampling within each participant file",
            "maximum_per_person_per_archive": args.window_limit,
            "seed": SEED,
            "all_windows_counted": True,
        },
        "no_skill_average_precision_reference": round(4 / 6, 4),
        "duration_variants": archive_reports,
        "deployment_status": "not_ready",
        "candidate_meets_exploratory_gate": False,
        "readiness_reasons": [
            "Only six identified participants are available in each raw archive (four with COPD and two controls); many windows do not increase the independent cohort size.",
            "All archives are from the same six people and are not independent external cohorts.",
            "The experiment uses public PPG signals rather than this proposed ring's calibrated red/infrared and motion pipeline.",
            "The original study's 14-person engineered-feature file has no participant IDs and its healthy-window count conflicts with the published per-person table, so it is excluded from this grouped evaluation.",
            "No external validation, calibration study, prospective collection, or clinical review has been completed.",
        ],
        "inference_enabled": False,
        "raw_data_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
        "independent_people": report["independent_participants_per_archive"],
        "cohort": report["cohort_per_archive"],
        "duration_results": {
            duration: {
                model: {
                    "roc_auc": metrics["participant_level_roc_auc"],
                    "average_precision": metrics["participant_level_average_precision"],
                    "balanced_accuracy": metrics["participant_level_balanced_accuracy"],
                }
                for model, metrics in details["models"].items()
            }
            for duration, details in archive_reports.items()
        },
        "deployment_status": report["deployment_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
