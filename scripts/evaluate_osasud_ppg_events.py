#!/usr/bin/env python3
"""Patient-held-out feasibility study for PPG-based breathing-event screening.

This evaluates annotated apnea/hypopnea event presence in 30-second windows.
It does not diagnose pulmonary disease or sleep apnea syndrome. The raw OSASUD
pickle is large, stays under ignored data/, and is never copied to model output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "osasud" / "dataset_OSAS.pickle"
REPORT_PATH = ROOT / "models" / "osasud_ppg_event_metrics.json"
CACHE_PATH = ROOT / "data" / "osasud" / "osasud_ppg_event_features_v1.npz"
WINDOW_SECONDS = 30
REQUIRED_COLUMNS = (
    "patient",
    "timestamp_datetime",
    "SpO2(%)",
    "PI(%)",
    "HR(bpm)",
    "RR(rpm)",
    "event",
    "signal_pleth",
)


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _summary(values: np.ndarray, prefix: str, out: dict[str, float]) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    out[f"{prefix}_coverage"] = float(finite.size / max(values.size, 1))
    if finite.size == 0:
        out.update({f"{prefix}_{name}": np.nan for name in ("mean", "std", "min", "max", "p10", "p90", "slope")})
        return
    out[f"{prefix}_mean"] = float(np.mean(finite))
    out[f"{prefix}_std"] = float(np.std(finite))
    out[f"{prefix}_min"] = float(np.min(finite))
    out[f"{prefix}_max"] = float(np.max(finite))
    out[f"{prefix}_p10"] = float(np.percentile(finite, 10))
    out[f"{prefix}_p90"] = float(np.percentile(finite, 90))
    if finite.size >= 2:
        positions = np.flatnonzero(np.isfinite(values)).astype(np.float64)
        out[f"{prefix}_slope"] = float(np.polyfit(positions, finite, 1)[0])
    else:
        out[f"{prefix}_slope"] = 0.0


def _waveform_features(rows: pd.Series) -> tuple[float, float, float, float, float, float, float]:
    amplitudes: list[float] = []
    ac_rms: list[float] = []
    derivative_rms: list[float] = []
    coverages = 0
    waveform_by_second: list[np.ndarray] = []
    for signal in rows:
        if signal is None:
            continue
        values = _finite_values(signal)
        if values.size < 40:
            waveform_by_second.append(np.full(80, np.nan, dtype=np.float64))
            continue
        coverages += 1
        if values.size == 80:
            waveform_by_second.append(values)
        else:
            source = np.linspace(0.0, 1.0, values.size)
            waveform_by_second.append(np.interp(np.linspace(0.0, 1.0, 80), source, values))
        centered = values - np.median(values)
        amplitudes.append(float(np.percentile(centered, 95) - np.percentile(centered, 5)))
        ac_rms.append(float(np.sqrt(np.mean(np.square(centered)))))
        derivative_rms.append(float(np.sqrt(np.mean(np.square(np.diff(centered))))))
    n = max(len(rows), 1)
    coverage = float(coverages / n)
    if not amplitudes:
        return np.nan, np.nan, np.nan, coverage, np.nan, np.nan, np.nan

    # Derive pulse timing from PPG, never from OSASUD's ECG-derived HR/RR.
    # Preserve missing spans and filter only contiguous valid runs so missing
    # seconds are not mistaken for rapid pulse intervals.
    if len(waveform_by_second) < n:
        waveform_by_second.extend([np.full(80, np.nan)] * (n - len(waveform_by_second)))
    raw = np.concatenate(waveform_by_second)
    finite = np.isfinite(raw)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
    sos = butter(2, (0.5, 5.0), btype="bandpass", fs=80, output="sos")
    intervals: list[float] = []
    for start, stop in zip(edges[::2], edges[1::2]):
        run = raw[start:stop]
        if run.size < 3 * 80:
            continue
        try:
            filtered = sosfiltfilt(sos, run - np.median(run))
        except ValueError:
            continue
        prominence = max(float(np.std(filtered)) * 0.12, 1e-8)
        peaks, _ = find_peaks(filtered, distance=24, prominence=prominence)
        intervals.extend((np.diff(peaks) / 80.0).tolist())
    intervals_array = np.asarray(intervals, dtype=np.float64)
    intervals_array = intervals_array[(intervals_array >= 60.0 / 200.0) & (intervals_array <= 60.0 / 30.0)]
    if intervals_array.size:
        pulse_rate = float(60.0 / np.median(intervals_array))
        ibi_cv = float(np.std(intervals_array) / max(np.mean(intervals_array), 1e-8))
        ibi_rmssd = float(np.sqrt(np.mean(np.square(np.diff(intervals_array))))) if intervals_array.size > 1 else np.nan
    else:
        pulse_rate = ibi_cv = ibi_rmssd = np.nan
    return (
        float(np.median(amplitudes)),
        float(np.median(ac_rms)),
        float(np.median(derivative_rms)),
        coverage,
        pulse_rate,
        ibi_cv,
        ibi_rmssd,
    )


def _build_windows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[str] = []
    skipped = 0
    event_counts: dict[str, int] = {}
    people = frame["patient"].astype(str).unique()
    for participant_index, (patient, subject) in enumerate(frame.groupby("patient", sort=False), start=1):
        subject = subject.sort_values("timestamp_datetime", kind="stable").reset_index(drop=True)
        events = subject["event"].fillna("NONE").astype(str).str.upper()
        for event, count in events.value_counts().items():
            event_counts[event] = event_counts.get(event, 0) + int(count)
        seconds = subject["timestamp_datetime"].astype("int64").to_numpy() // 1_000_000_000
        breaks = np.flatnonzero(np.diff(seconds) != 1) + 1
        starts = np.r_[0, breaks]
        ends = np.r_[breaks, len(subject)]
        for start, end in zip(starts, ends):
            block = subject.iloc[start:end]
            block_events = events.iloc[start:end].to_numpy()
            for offset in range(0, len(block) - WINDOW_SECONDS + 1, WINDOW_SECONDS):
                sample = block.iloc[offset : offset + WINDOW_SECONDS]
                sample_events = block_events[offset : offset + WINDOW_SECONDS]
                # Require at least 80% valid PPG seconds; missingness is not a
                # hidden target and low-quality windows are excluded explicitly.
                ppg = sample["signal_pleth"]
                waveform = _waveform_features(ppg)
                ppg_coverage = waveform[3]
                if ppg_coverage < 0.8:
                    skipped += 1
                    continue
                features: dict[str, float] = {}
                for column, prefix in (("SpO2(%)", "spo2"), ("PI(%)", "perfusion")):
                    values = pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    if prefix == "spo2":
                        values[(values < 50) | (values > 100)] = np.nan
                    elif prefix == "pulse_rate":
                        values[(values < 20) | (values > 240)] = np.nan
                    elif prefix == "perfusion":
                        values[(values < 0) | (values > 100)] = np.nan
                    _summary(values, prefix, features)
                # A separate comparison track uses these ring-like scalar
                # outputs, but their reference sources are hospital ECG and
                # thoracic impedance rather than a ring. They never enter the
                # strict PPG-only candidate.
                for column, prefix in (("HR(bpm)", "reference_hr"), ("RR(rpm)", "reference_rr")):
                    values = pd.to_numeric(sample[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    lower, upper = (20, 240) if prefix.endswith("hr") else (4, 60)
                    values[(values < lower) | (values > upper)] = np.nan
                    _summary(values, prefix, features)
                features["ppg_amplitude_p2p_median"] = waveform[0]
                features["ppg_ac_rms_median"] = waveform[1]
                features["ppg_derivative_rms_median"] = waveform[2]
                features["ppg_valid_seconds"] = waveform[3]
                features["ppg_derived_pulse_rate_bpm"] = waveform[4]
                features["ppg_ibi_cv"] = waveform[5]
                features["ppg_ibi_rmssd_seconds"] = waveform[6]
                rows.append(features)
                labels.append(int(np.isin(sample_events, ["HYPOPNEA", "APNEA-CENTRAL", "APNEA-OBSTRUCTIVE", "APNEA-MIXED"]).any()))
                groups.append(str(patient))
        if participant_index % 1 == 0:
            print(f"Processed {participant_index}/{len(people)} participants into quality-checked PPG windows...", flush=True)
    if not rows:
        raise RuntimeError("No quality-valid 30-second windows were built from OSASUD.")
    matrix = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    group_array = np.asarray(groups)
    return matrix, target, group_array, list(pd.DataFrame(rows).columns), {
        "source_rows": int(len(frame)),
        "source_participants": int(len(people)),
        "windows": int(len(target)),
        "positive_windows": int(target.sum()),
        "positive_prevalence": float(target.mean()),
        "excluded_low_ppg_quality_windows": int(skipped),
        "annotation_seconds_by_label": dict(sorted(event_counts.items())),
        "seconds_per_window": WINDOW_SECONDS,
        "participants_with_windows": int(len(np.unique(group_array))),
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
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(n_estimators=250, min_samples_leaf=20, max_features=0.75, class_weight="balanced", n_jobs=-1, random_state=2026),
        ),
    }


def _weighted_metrics(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "roc_auc_equal_participant_weight": None,
        "average_precision_equal_participant_weight": float(average_precision_score(y, scores, sample_weight=weights)),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc_equal_participant_weight"] = float(roc_auc_score(y, scores, sample_weight=weights))
    result["roc_auc_pooled_windows"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
    result["average_precision_pooled_windows"] = float(average_precision_score(y, scores))
    result["threshold_0_5_sensitivity_equal_participant_weight"] = float(np.average((scores >= 0.5)[y == 1], weights=weights[y == 1])) if np.any(y == 1) else None
    result["threshold_0_5_specificity_equal_participant_weight"] = float(np.average((scores < 0.5)[y == 0], weights=weights[y == 0])) if np.any(y == 0) else None
    result["threshold_0_5_balanced_accuracy_pooled_windows"] = float(balanced_accuracy_score(y, scores >= 0.5))
    return result


def _evaluate_features(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[dict[str, Any], float]:
    n_groups = len(np.unique(groups))
    if n_groups < 5 or len(np.unique(y)) < 2:
        raise RuntimeError("At least five participants and both outcome classes are required for grouped evaluation.")
    splitter = GroupKFold(n_splits=min(5, n_groups))
    participant_sizes = pd.Series(groups).value_counts().to_dict()
    weights = np.asarray([1.0 / participant_sizes[group] for group in groups], dtype=np.float64)
    # Normalize so weighted AP's positive prevalence remains interpretable.
    weights *= len(weights) / weights.sum()
    model_reports: dict[str, Any] = {}
    for name, model in _models().items():
        oof = np.full(len(y), np.nan, dtype=np.float64)
        folds: list[dict[str, Any]] = []
        for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
            if len(np.unique(y[train])) < 2:
                continue
            fit_options: dict[str, Any] = {}
            train_sizes = pd.Series(groups[train]).value_counts().to_dict()
            train_weights = np.asarray([1.0 / train_sizes[group] for group in groups[train]], dtype=np.float64)
            train_weights *= len(train_weights) / train_weights.sum()
            estimator_name = model.steps[-1][0]
            if name == "hist_gradient_boosting":
                train_weights *= compute_sample_weight("balanced", y[train])
            fit_options[f"{estimator_name}__sample_weight"] = train_weights
            model.fit(X[train], y[train], **fit_options)
            prediction = model.predict_proba(X[test])[:, 1]
            oof[test] = prediction
            fold_weights = weights[test]
            folds.append({
                "fold": fold,
                "held_out_participants": int(len(np.unique(groups[test]))),
                "windows": int(len(test)),
                "positive_windows": int(y[test].sum()),
                "equal_participant_average_precision": float(average_precision_score(y[test], prediction, sample_weight=fold_weights)),
                "equal_participant_roc_auc": float(roc_auc_score(y[test], prediction, sample_weight=fold_weights)) if len(np.unique(y[test])) == 2 else None,
            })
        valid = np.isfinite(oof)
        metric = _weighted_metrics(y[valid], oof[valid], weights[valid])
        model_reports[name] = {"metrics": metric, "folds": folds}
    return model_reports, float(np.average(y, weights=weights))


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_names: list[str], counts: dict[str, Any]) -> dict[str, Any]:
    feature_sets = {
        "strict_ppg_only": [name for name in feature_names if name.startswith(("spo2_", "perfusion_", "ppg_"))],
        "expanded_ring_like_vitals": [name for name in feature_names if name.startswith(("spo2_", "perfusion_", "ppg_", "reference_hr_", "reference_rr_"))],
    }


def _load_or_build_windows(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    stat = data_path.stat()
    if CACHE_PATH.is_file():
        try:
            with np.load(CACHE_PATH, allow_pickle=False) as cache:
                metadata = json.loads(cache["metadata"].tobytes().decode("utf-8"))
                if metadata.get("source_size") == stat.st_size and metadata.get("source_mtime_ns") == stat.st_mtime_ns:
                    print("Using cached aggregate PPG window features; the source pickle does not need to be loaded again.", flush=True)
                    return (
                        cache["X"].copy(), cache["y"].copy(), cache["groups"].copy(),
                        metadata["feature_names"], metadata["counts"],
                    )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            print("The local feature cache is invalid; rebuilding it from the source dataset.", flush=True)

    print(f"Loading {data_path} (large public dataset; the first run may take several minutes)...", flush=True)
    frame = pd.read_pickle(data_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise SystemExit(f"OSASUD is missing required columns: {missing}")
    print(f"Loaded {len(frame):,} one-second rows across {frame['patient'].nunique()} participants.", flush=True)
    X, y, raw_groups, feature_names, counts = _build_windows(frame)
    del frame
    _, group_codes = np.unique(raw_groups, return_inverse=True)
    group_codes = group_codes.astype(np.int16, copy=False)
    metadata = {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "feature_names": feature_names,
        "counts": counts,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X=X,
        y=y,
        groups=group_codes,
        metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
    )
    print(f"Built {len(y):,} quality-valid windows; event prevalence {y.mean():.3f}. Cached derived features locally.", flush=True)
    return X, y, group_codes, feature_names, counts
    track_results: dict[str, Any] = {}
    candidates: list[tuple[float, float, str, str, dict[str, Any]]] = []
    for track_name, names in feature_sets.items():
        indexes = [feature_names.index(name) for name in names]
        model_reports, prevalence = _evaluate_features(X[:, indexes], y, groups)
        track_results[track_name] = {
            "input_features": names,
            "naive_average_precision_reference_equal_participant_weight": prevalence,
            "models": model_reports,
        }
        for model_name, model_report in model_reports.items():
            metrics = model_report["metrics"]
            ap = float(metrics["average_precision_equal_participant_weight"] or 0)
            auc = float(metrics["roc_auc_equal_participant_weight"] or 0)
            candidates.append((ap, auc, track_name, model_name, metrics))
    strict_candidates = [item for item in candidates if item[2] == "strict_ppg_only"]
    ap, auc, best_track, best_name, best = max(strict_candidates, key=lambda item: (item[0], item[1]))
    prevalence = track_results["strict_ppg_only"]["naive_average_precision_reference_equal_participant_weight"]
    return {
        "model_version": "plumoring-osasud-ppg-event-feasibility-0.1",
        "dataset": "OSASUD · stroke-unit overnight PSG and finger PPG",
        "dataset_doi": "10.1038/s41597-022-01272-y",
        "dataset_url": "https://springernature.figshare.com/articles/dataset/Main_dataset_file/19214289",
        "task": "research-only detection of any physician-annotated apnea or hypopnea in a 30-second overnight window among hospitalized stroke-unit patients",
        "input_channels": {
            "strict_ppg_only": ["finger PPG waveform", "PPG-derived pulse timing and interval variability", "PPG-derived oxygen saturation", "PPG-derived perfusion index"],
            "expanded_ring_like_vitals": ["strict PPG-only inputs", "reference heart rate from hospital ECG", "reference respiratory rate from hospital thoracic impedance"],
        },
        "excluded_channels": ["ECG waveforms", "PSG airflow", "thoracic/abdominal effort", "snoring/audio", "body position", "labels or event metadata"],
        "secondary_track_sensor_source_note": "The expanded scalar-vitals comparison adds hospital ECG-derived HR and thoracic-impedance-derived RR; it is not a ring-only model and is not eligible for ring inference.",
        "feature_names": feature_names,
        "split": "5-fold GroupKFold by participant; no participant appears in both train and held-out fold",
        "counts": counts,
        "naive_average_precision_reference_equal_participant_weight": prevalence,
        "best_candidate": {"feature_set": best_track, "model": best_name},
        "best_candidate_metrics": best,
        "feature_set_results": track_results,
        "exploratory_gate": {
            "criteria": "strict PPG-only candidate: equal-participant-weight average precision >= 1.25x positive prevalence and equal-participant-weight ROC-AUC >= 0.60",
            "passed": bool(ap >= prevalence * 1.25 and auc >= 0.60),
            "patient_facing_or_diagnostic": False,
            "selection_limitation": "The best of several baseline models is selected on the same grouped out-of-fold comparison; its reported gate is exploratory and optimistic. Independent external validation is still required.",
        },
        "deployment_status": "not_deployed_pending_independent_external_validation",
        "clinical_limitations": [
            "This is event detection during overnight monitoring, not diagnosis of obstructive sleep apnea syndrome.",
            "The 30-person cohort consists of hospitalized patients admitted for suspected cerebrovascular events; it is not representative of a community ring population.",
            "Signals were captured by clinical finger PPG hardware, not the proposed PlumoRing device.",
            "An exploratory grouped result does not establish clinical utility or validate a continuous ring alert.",
            "This dataset does not support detecting COPD, asthma, pneumonia, bronchitis, infection, or daytime pulmonary disease.",
        ],
        "reproduction": "python scripts/evaluate_osasud_ppg_events.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Missing OSASUD pickle: {args.data}")
    X, y, groups, feature_names, counts = _load_or_build_windows(args.data)
    report = evaluate(X, y, groups, feature_names, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "counts": report["counts"],
        "best_candidate": report["best_candidate"],
        "best_candidate_metrics": report["best_candidate_metrics"],
        "gate": report["exploratory_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
