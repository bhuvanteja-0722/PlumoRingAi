#!/usr/bin/env python3
"""Evaluate PPG-only respiratory-rate estimation on BIDMC.

The predictors use only the PLETH pulse-oximeter waveform. The target is derived
from two independent manual breath-annotation streams. Impedance respiration,
monitor numerics, ECG and demographics are never predictors. Results are
aggregate-only; this ICU pulse-oximeter study is not ring or disease validation.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, hilbert, periodogram, sosfiltfilt
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "bidmc" / "bidmc.zip"
OUTPUT = ROOT / "models" / "bidmc_ppg_respiratory_rate_metrics.json"
FS = 125.0
WINDOW_SECONDS = 30
STEP_SECONDS = 30


def _spectrum(x: np.ndarray, fs: float, prefix: str) -> tuple[dict[str, float], float]:
    if len(x) < 32 or not np.isfinite(x).all() or np.std(x) < 1e-9:
        return {}, math.nan
    centered = x - np.mean(x)
    nfft = max(2048, 2 ** int(math.ceil(math.log2(len(centered)))))
    freq, power = periodogram(centered, fs=fs, nfft=nfft)
    mask = (freq >= 0.08) & (freq <= 0.65)
    if not mask.any():
        return {}, math.nan
    fb, pb = freq[mask], power[mask]
    total = float(np.sum(pb))
    if not math.isfinite(total) or total <= 1e-15:
        return {}, math.nan
    peak = int(np.argmax(pb))
    norm = pb / total
    values = {
        f"{prefix}_peak_rate": float(fb[peak] * 60),
        f"{prefix}_peak_fraction": float(pb[peak] / total),
        f"{prefix}_band_power": total,
        f"{prefix}_entropy": float(-np.sum(norm * np.log(norm + 1e-12)) / max(math.log(len(norm)), 1e-12)),
        f"{prefix}_low_fraction": float(np.sum(pb[fb < 0.2]) / total),
        f"{prefix}_mid_fraction": float(np.sum(pb[(fb >= 0.2) & (fb < 0.35)]) / total),
        f"{prefix}_high_fraction": float(np.sum(pb[fb >= 0.35]) / total),
    }
    return values, float(fb[peak] * 60)


def ppg_features(signal: np.ndarray, fs: float = FS) -> tuple[dict[str, float], dict[str, float]]:
    """Summarize 30 s PPG morphology plus respiratory modulation spectra."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(x) < int(fs * 25) or np.mean(np.isfinite(x)) < 0.99:
        return {}, {}
    x = x[np.isfinite(x)]
    x = x - np.median(x)
    sd = float(np.std(x))
    if sd < 1e-9:
        return {}, {}
    x = x / sd
    features: dict[str, float] = {
        "ppg_std": float(np.std(x)),
        "ppg_range": float(np.percentile(x, 95) - np.percentile(x, 5)),
        "ppg_mad": float(np.median(np.abs(x - np.median(x)))),
        "ppg_line_length": float(np.mean(np.abs(np.diff(x)))),
    }
    direct, direct_peak = _spectrum(x, fs, "ppg_baseline")
    features.update(direct)

    try:
        pulse = sosfiltfilt(butter(3, [0.5, min(4.0, fs * 0.45)], btype="bandpass", fs=fs, output="sos"), x)
        features["ppg_pulse_ac_rms"] = float(np.sqrt(np.mean(pulse**2)))
        envelope = np.abs(hilbert(pulse))
        resp_env = sosfiltfilt(butter(3, [0.08, 0.65], btype="bandpass", fs=fs, output="sos"), envelope)
        env, env_peak = _spectrum(resp_env, fs, "ppg_envelope")
        features.update(env)
    except ValueError:
        pulse = np.array([], dtype=np.float64)
        env_peak = math.nan

    pulse_peak, interval_peak = math.nan, math.nan
    if pulse.size:
        beat_indices, _ = find_peaks(pulse, distance=int(fs * 0.30), prominence=max(0.15 * float(np.std(pulse)), 0.02))
        if len(beat_indices) >= 6:
            beat_times = beat_indices / fs
            amplitude = pulse[beat_indices]
            ibi = np.diff(beat_times)
            beat_mid = (beat_times[1:] + beat_times[:-1]) / 2
            grid = np.arange(0, min((len(x) - 1) / fs, 30.0), 0.25)
            if len(ibi) >= 4 and len(grid) >= 40:
                amp_grid = np.interp(grid, beat_times, amplitude)
                ibi_grid = np.interp(grid, beat_mid, ibi)
                amp_filtered = sosfiltfilt(butter(2, [0.08, 0.65], btype="bandpass", fs=4.0, output="sos"), amp_grid)
                ibi_filtered = sosfiltfilt(butter(2, [0.08, 0.65], btype="bandpass", fs=4.0, output="sos"), ibi_grid)
                amp_spec, pulse_peak = _spectrum(amp_filtered, 4.0, "ppg_amplitude_mod")
                ibi_spec, interval_peak = _spectrum(ibi_filtered, 4.0, "ppg_interval_mod")
                features.update(amp_spec)
                features.update(ibi_spec)
                features["pulse_rate_mean"] = float(60 / np.median(ibi))
                features["pulse_rate_std"] = float(np.std(60 / ibi))
                features["pulse_amp_cv"] = float(np.std(amplitude) / (abs(np.mean(amplitude)) + 1e-9))
                features["pulse_count"] = float(len(beat_indices))
    direct_candidates = [rate for rate in (direct_peak, env_peak, pulse_peak, interval_peak) if math.isfinite(rate)]
    direct_summary = {
        "raw_baseline_spectral_peak": direct_peak,
        "pulse_envelope_modulation_peak": env_peak,
        "pulse_amplitude_modulation_peak": pulse_peak,
        "pulse_interval_modulation_peak": interval_peak,
    }
    if direct_candidates:
        direct_summary["best_individual_spectral_peak"] = float(np.median(direct_candidates))
    return features, direct_summary


def _rate_from_annotations(values: pd.Series, start: float, end: float) -> float:
    times = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64) / FS
    local = times[(times >= start) & (times < end)]
    if len(local) < 4:
        return math.nan
    span = float(local[-1] - local[0])
    if span <= 0:
        return math.nan
    return float(60 * (len(local) - 1) / span)


def load_windows(archive_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records = 0
    skipped_records = 0
    annotator_r1: list[float] = []
    annotator_r2: list[float] = []
    base = "bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/"
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith("_Signals.csv"))
        for signal_name in names:
            record = signal_name.rsplit("/", 1)[-1].replace("_Signals.csv", "")
            breath_name = f"{base}{record}_Breaths.csv"
            try:
                signals = pd.read_csv(io.BytesIO(archive.read(signal_name)))
                breaths = pd.read_csv(io.BytesIO(archive.read(breath_name)))
            except (KeyError, ValueError, OSError):
                skipped_records += 1
                continue
            signals.columns = signals.columns.str.strip()
            breaths.columns = breaths.columns.str.strip()
            if not {"Time [s]", "PLETH"}.issubset(signals.columns) or len(signals) < 1000:
                skipped_records += 1
                continue
            ppg = pd.to_numeric(signals["PLETH"], errors="coerce").to_numpy(dtype=np.float64)
            valid_rate1 = breaths.iloc[:, 0].count()
            valid_rate2 = breaths.iloc[:, 1].count()
            if len(ppg) < int(FS * WINDOW_SECONDS):
                skipped_records += 1
                continue
            records += 1
            for start in np.arange(0, len(ppg) / FS - WINDOW_SECONDS + 1e-6, STEP_SECONDS):
                end = float(start + WINDOW_SECONDS)
                left, right = int(round(start * FS)), int(round(end * FS))
                segment = ppg[left:right]
                if len(segment) < int(FS * (WINDOW_SECONDS - 1)) or np.mean(np.isfinite(segment)) < 0.99:
                    continue
                rate1 = _rate_from_annotations(breaths.iloc[:, 0], float(start), end)
                rate2 = _rate_from_annotations(breaths.iloc[:, 1], float(start), end)
                if not (math.isfinite(rate1) and math.isfinite(rate2)):
                    continue
                target = float((rate1 + rate2) / 2)
                if not 4 <= target <= 50:
                    continue
                features, spectral = ppg_features(segment)
                if not features:
                    continue
                annotator_r1.append(rate1)
                annotator_r2.append(rate2)
                rows.append({"participant": record, "start_seconds": int(start), "rr": target, "rr_annotator_1": rate1, "rr_annotator_2": rate2, **features, **{f"direct_{k}": v for k, v in spectral.items()}})
    frame = pd.DataFrame(rows)
    counts = {
        "recordings_loaded": records,
        "recordings_skipped": skipped_records,
        "participants": int(frame["participant"].nunique()) if len(frame) else 0,
        "windows": int(len(frame)),
        "window_seconds": WINDOW_SECONDS,
        "window_step_seconds": STEP_SECONDS,
        "manual_annotation_windows_with_both_readers": int(len(frame)),
        "annotator_mean_absolute_difference_bpm": float(np.mean(np.abs(np.asarray(annotator_r1) - np.asarray(annotator_r2)))) if annotator_r1 else None,
        "annotator_pearson_r": float(pearsonr(annotator_r1, annotator_r2).statistic) if len(annotator_r1) > 2 else None,
        "target_rr_range_bpm": [float(frame.rr.min()), float(frame.rr.max())] if len(frame) else None,
        "target_rr_median_bpm": float(frame.rr.median()) if len(frame) else None,
    }
    return frame, counts


def _pipeline(model: Any) -> Any:
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)


def _models() -> dict[str, Any]:
    return {
        "participant_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=10)),
        "extra_trees": _pipeline(ExtraTreesRegressor(n_estimators=350, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=29)),
        "random_forest": _pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.8, n_jobs=-1, random_state=29)),
        "hist_gradient_boosting": _pipeline(HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=7, min_samples_leaf=15, l2_regularization=5.0, random_state=29)),
    }


def _metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    good = np.isfinite(y) & np.isfinite(pred)
    y, pred, groups = y[good], pred[good], groups[good]
    subject_mae = [float(mean_absolute_error(y[groups == group], pred[groups == group])) for group in np.unique(groups)]
    rng = np.random.default_rng(20261127)
    values = np.asarray(subject_mae)
    if len(values) > 1:
        samples = values[rng.integers(0, len(values), size=(3000, len(values)))].mean(axis=1)
        ci = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    else:
        ci = None
    return {
        "windows": int(len(y)),
        "participants": int(len(np.unique(groups))),
        "mae_bpm": float(mean_absolute_error(y, pred)),
        "participant_macro_mae_bpm": float(np.mean(subject_mae)),
        "participant_macro_mae_95_ci_bpm": ci,
        "rmse_bpm": float(np.sqrt(mean_squared_error(y, pred))),
        "bias_bpm": float(np.mean(pred - y)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
        "pearson_r": float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else None,
        "spearman_r": float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else None,
    }


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    excluded = {"participant", "start_seconds", "rr", "rr_annotator_1", "rr_annotator_2", "direct_raw_baseline_spectral_peak", "direct_pulse_envelope_modulation_peak", "direct_pulse_amplitude_modulation_peak", "direct_pulse_interval_modulation_peak", "direct_best_individual_spectral_peak"}
    feature_names = [name for name in frame.columns if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])]
    X = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame.rr.to_numpy(dtype=np.float64)
    groups = frame.participant.to_numpy()
    split = GroupKFold(n_splits=5)
    all_results: dict[str, Any] = {}
    for name, template in _models().items():
        pred = np.full(len(y), np.nan)
        for train, test in split.split(X, y, groups):
            model = _models()[name]
            model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])
        all_results[name] = _metrics(y, pred, groups)
    for col in ("direct_raw_baseline_spectral_peak", "direct_pulse_envelope_modulation_peak", "direct_pulse_amplitude_modulation_peak", "direct_pulse_interval_modulation_peak", "direct_best_individual_spectral_peak"):
        all_results[col.removeprefix("direct_")] = _metrics(y, frame[col].to_numpy(dtype=float), groups)

    candidates = tuple(name for name in _models() if name != "participant_mean")
    outer_pred = np.full(len(y), np.nan)
    selections: list[dict[str, Any]] = []
    for outer_id, (train, test) in enumerate(split.split(X, y, groups), start=1):
        train_groups = groups[train]
        inner = GroupKFold(n_splits=min(3, len(np.unique(train_groups))))
        inner_scores: dict[str, float] = {}
        for name in candidates:
            inner_pred = np.full(len(train), np.nan)
            for inner_train, inner_test in inner.split(X[train], y[train], train_groups):
                model = _models()[name]
                model.fit(X[train][inner_train], y[train][inner_train])
                inner_pred[inner_test] = model.predict(X[train][inner_test])
            macro = _metrics(y[train], inner_pred, train_groups)["participant_macro_mae_bpm"]
            inner_scores[name] = macro
        selected = min(inner_scores, key=inner_scores.get)
        model = _models()[selected]
        model.fit(X[train], y[train])
        outer_pred[test] = model.predict(X[test])
        selections.append({"outer_fold": outer_id, "selected_model_inside_training_participants": selected, "inner_grouped_macro_mae": inner_scores, "held_out_participants": int(len(np.unique(groups[test]))), "held_out_windows": int(len(test))})

    return {
        "validation": "Nested five-fold GroupKFold by participant; all windows from a patient stay together. Inner three-fold grouped evaluation selects model family, and outer participants are scored once.",
        "input_features": feature_names,
        "participant_held_out_model_comparison": all_results,
        "nested_model_selection": {**_metrics(y, outer_pred, groups), "selected_models_by_outer_fold": selections},
        "limitations": [
            "BIDMC is 53 short recordings from critically ill hospital patients, not a ring cohort and not a pulmonary diagnosis study.",
            "The target is derived from two manual breath annotations made from impedance respiration; impedance is not a model input.",
            "Thirty-second local breathing-rate estimates are noisy and annotation agreement limits the achievable precision.",
            "A PPG pulse-oximeter experiment does not establish performance on the proposed finger ring or on community patients.",
            "This estimates respiratory rate; it does not classify COPD, asthma, pneumonia, infection, or exacerbation.",
        ],
    }


def external_ring_transfer(frame: pd.DataFrame) -> dict[str, Any]:
    """Train on BIDMC PPG features only; evaluate unchanged on ring IR features."""
    ring_path = ROOT / "data" / "healthring" / "ring_vitals_features_v1.csv"
    if not ring_path.is_file():
        return {"status": "unavailable", "reason": "The local HealthRing feature cache was not found."}
    ring = pd.read_csv(ring_path)
    if "cache_version" in ring:
        ring = ring.drop(columns=["cache_version"])
    # Only shared morphology and raw-spectrum features computable from one ring
    # infrared PPG channel are eligible for this device-transfer experiment.
    ring_std = pd.to_numeric(ring.get("ir_std"), errors="coerce").replace(0, np.nan)
    normalized = pd.DataFrame({
        "_ir_range_norm": pd.to_numeric(ring.get("ir_range"), errors="coerce") / ring_std,
        "_ir_mad_norm": pd.to_numeric(ring.get("ir_median_abs_dev"), errors="coerce") / ring_std,
        "_ir_pulse_ac_norm": pd.to_numeric(ring.get("ir_pulse_ac_rms"), errors="coerce") / ring_std,
    }, index=ring.index)
    ring = pd.concat([ring, normalized], axis=1)
    transfer_map = {
        "ppg_range": "_ir_range_norm",
        "ppg_mad": "_ir_mad_norm",
        "ppg_pulse_ac_rms": "_ir_pulse_ac_norm",
        "ppg_baseline_peak_rate": "ir_peak_rate",
        "ppg_baseline_peak_fraction": "ir_peak_power_fraction",
        "ppg_baseline_entropy": "ir_spectral_entropy",
        "ppg_baseline_low_fraction": "ir_band_1_fraction",
        "ppg_baseline_mid_fraction": "ir_band_2_fraction",
        "ppg_baseline_high_fraction": "ir_band_3_fraction",
    }
    transfer_features = [name for name in transfer_map if name in frame.columns and transfer_map[name] in ring]
    subset = ring[(ring.get("ring") == "ring1") & (ring.get("cohort") == "controlled_daily")].copy()
    subset["rr"] = pd.to_numeric(subset.get("rr"), errors="coerce")
    subset["participant"] = subset.get("participant").astype(str)
    subset = subset[subset.rr.between(4, 40)]
    if len(subset) < 20 or subset.participant.nunique() < 5 or not transfer_features:
        return {"status": "insufficient_data", "test_windows": int(len(subset)), "test_participants": int(subset.participant.nunique())}
    X_train = frame[transfer_features].to_numpy(dtype=np.float64)
    y_train = frame.rr.to_numpy(dtype=np.float64)
    X_test = subset[[transfer_map[name] for name in transfer_features]].to_numpy(dtype=np.float64)
    y_test = subset.rr.to_numpy(dtype=np.float64)
    groups = subset.participant.to_numpy()
    # Select the transfer model using only the matching source-domain features
    # and participant-held-out folds; the ring test cohort stays untouched.
    source_frame = frame[transfer_features + ["rr", "participant"]]
    source_X = source_frame[transfer_features].to_numpy(dtype=np.float64)
    source_y = source_frame.rr.to_numpy(dtype=np.float64)
    source_groups = source_frame.participant.to_numpy()
    source_cv = GroupKFold(n_splits=5)
    eligible: dict[str, float] = {}
    for name in ("ridge", "extra_trees", "random_forest", "hist_gradient_boosting"):
        source_oof = np.full(len(source_y), np.nan)
        for train, test in source_cv.split(source_X, source_y, source_groups):
            candidate = _models()[name]
            candidate.fit(source_X[train], source_y[train])
            source_oof[test] = candidate.predict(source_X[test])
        eligible[name] = _metrics(source_y, source_oof, source_groups)["participant_macro_mae_bpm"]
    selected = min(eligible, key=eligible.get)
    model = _models()[selected]
    model.fit(source_X, source_y)
    prediction = model.predict(X_test)
    # A source-domain mean baseline is also evaluated on the ring test cohort.
    baseline = float(np.mean(y_train))
    test = _metrics(y_test, prediction, groups)
    test["source_selected_model"] = selected
    test["participant_mean_baseline"] = _metrics(y_test, np.full(len(y_test), baseline), groups)
    return {
        "status": "evaluated",
        "train_dataset": "BIDMC PPG / ICU patients",
        "external_test_dataset": "HealthRing Ring 1 infrared PPG / healthy-volunteer reference RR",
        "train_participants": int(frame.participant.nunique()),
        "train_windows": int(len(frame)),
        "shared_ppg_features": transfer_features,
        "source_domain_model_selection_macro_mae_bpm": eligible,
        "test_participants": int(subset.participant.nunique()),
        "test_windows": int(len(subset)),
        "input_policy": "Source-domain model trained only on PPG waveform morphology/spectral features; no impedance, ECG, monitor numerics, or ring reference signals used as predictors. The Ring 1 cohort is an untouched external domain test.",
        "performance": test,
        "limitations": [
            "This is a cross-device and cross-population transfer test, not disease detection or ring hardware validation.",
            "BIDMC pulse-oximeter PPG and custom HealthRing infrared PPG have different sensors, study protocols, and populations.",
            "The HealthRing data source is locally recovered and partial; its official archive lacks a usable ZIP directory.",
        ],
    }


def build_report(frame: pd.DataFrame, counts: dict[str, Any]) -> dict[str, Any]:
    results = evaluate(frame)
    return {
        "model_version": "plumoring-bidmc-ppg-rr-0.1",
        "dataset": "BIDMC PPG and Respiration Dataset 1.0.0",
        "dataset_doi": "10.13026/C2208R",
        "source": "https://physionet.org/content/bidmc/1.0.0/",
        "license": "Open Data Commons Attribution License v1.0",
        "task": "Estimate local respiratory rate in 30-second PPG windows from manual breath annotations.",
        "predictor_policy": "Only the 125 Hz PLETH pulse-oximeter waveform and transforms of that waveform are model inputs. No RESP waveform, ECG, monitor numerics, SpO2, participant metadata, or labels are predictors.",
        "target_policy": "Mean of two independent manual breath annotation rates per 30-second window; annotation times are in signal-sample coordinates at 125 Hz.",
        "counts": counts,
        "results": results,
        "cross_device_healthring_transfer": external_ring_transfer(frame),
        "deployment_status": "research_only_no_disease_classifier",
        "reproduction": "python scripts/evaluate_bidmc_ppg_respiratory_rate.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"BIDMC archive not found: {args.archive}")
    print("Extracting PPG-only 30-second windows and dual-reader breath targets...", flush=True)
    frame, counts = load_windows(args.archive)
    if frame.empty or counts["participants"] < 5:
        raise SystemExit("Insufficient BIDMC records were readable for participant-held-out evaluation.")
    print(f"Evaluating {counts['windows']} windows from {counts['participants']} participants with nested grouped model selection...", flush=True)
    report = build_report(frame, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": counts, "nested": report["results"]["nested_model_selection"], "transfer": report["cross_device_healthring_transfer"]}, indent=2))
    print(f"Wrote aggregate-only report: {args.output}")


if __name__ == "__main__":
    main()
