#!/usr/bin/env python3
"""Evaluate ring-only respiratory-rate and oxygenation estimates on HealthRing.

All model inputs are derived from the ring's infrared/red PPG and accelerometer.
Reference respiration, reference SpO2, ECG, oximeter BVP, and metadata labels are
never used as predictors. The report contains aggregate metrics only.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "healthring" / "RingDatasetV2.1_submission.zip"
CACHE = ROOT / "data" / "healthring" / "ring_vitals_features_v1.csv"
CACHE_META = ROOT / "data" / "healthring" / "ring_vitals_features_v1.meta.json"
REPORT = ROOT / "models" / "healthring_ring_vitals_metrics.json"
CACHE_VERSION = "healthring-ring-features-v1"


def _number(value: Any) -> float:
    try:
        if value is None or isinstance(value, (list, tuple, dict, np.ndarray)):
            return math.nan
        x = float(value)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _signal(value: Any) -> np.ndarray | None:
    try:
        x = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if x.size < 128:
        return None
    good = np.isfinite(x)
    if good.mean() < 0.98:
        return None
    x = x[good]
    if x.size < 128 or float(np.std(x)) < 1e-10:
        return None
    return x


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray | None:
    if not math.isfinite(fs) or fs <= 2 * high or len(x) < max(64, int(fs * 3)):
        return None
    try:
        return sosfiltfilt(butter(3, [low, high], btype="bandpass", fs=fs, output="sos"), x)
    except ValueError:
        return None


def _spectrum_features(x: np.ndarray, fs: float, prefix: str, low: float, high: float) -> dict[str, float]:
    f: dict[str, float] = {}
    x = x - np.mean(x)
    freq, power = periodogram(x, fs=fs, nfft=max(2048, 2 ** int(math.ceil(math.log2(len(x))))))
    band = (freq >= low) & (freq <= high)
    if not band.any():
        return f
    fb, pb = freq[band], power[band]
    peak = int(np.argmax(pb))
    total = float(np.sum(pb))
    eps = 1e-12
    f[f"{prefix}_peak_hz"] = float(fb[peak])
    f[f"{prefix}_peak_rate"] = float(fb[peak] * 60.0)
    f[f"{prefix}_peak_power_fraction"] = float(pb[peak] / max(total, eps))
    f[f"{prefix}_band_power"] = total
    norm = pb / max(total, eps)
    f[f"{prefix}_spectral_entropy"] = float(-np.sum(norm * np.log(norm + eps)) / max(math.log(len(norm)), eps))
    for i, (a, b) in enumerate(((low, 0.2), (0.2, 0.35), (0.35, high)), start=1):
        mask = (fb >= a) & (fb < b if b < high else fb <= b)
        f[f"{prefix}_band_{i}_fraction"] = float(np.sum(pb[mask]) / max(total, eps))
    return f


def _channel_features(row: pd.Series, key: str, fs: float, *, respiration: bool = False) -> dict[str, float]:
    x = _signal(row.get(key))
    if x is None:
        return {}
    short = key.replace("-raw", "").replace("-filtered-rr", "_rr").replace("-", "_")
    f: dict[str, float] = {
        f"{short}_mean": float(np.mean(x)),
        f"{short}_std": float(np.std(x)),
        f"{short}_range": float(np.percentile(x, 95) - np.percentile(x, 5)),
        f"{short}_median_abs_dev": float(np.median(np.abs(x - np.median(x)))),
    }
    ac = _bandpass(x, fs, 0.5, min(4.0, fs * 0.45))
    if ac is not None:
        f[f"{short}_pulse_ac_rms"] = float(np.sqrt(np.mean(ac**2)))
    if respiration:
        f.update(_spectrum_features(x, fs, short, 0.08, 0.65))
    else:
        f.update(_spectrum_features(x, fs, short, 0.08, 0.7))
        slow = _bandpass(x, fs, 0.08, 0.65)
        if slow is not None:
            f.update(_spectrum_features(slow, fs, f"{short}_slow", 0.08, 0.65))
    return f


def _features(row: pd.Series) -> tuple[dict[str, float], float]:
    fs = _number(row.get("fs"))
    if not math.isfinite(fs) or fs < 20:
        fs = 100.0
    features: dict[str, float] = {}
    for channel in ("ir-raw", "red-raw"):
        features.update(_channel_features(row, channel, fs))
    for channel in ("ir-filtered-rr", "red-filtered-rr"):
        features.update(_channel_features(row, channel, fs, respiration=True))
    axes = []
    for axis in ("ax-raw", "ay-raw", "az-raw"):
        x = _signal(row.get(axis))
        if x is None:
            continue
        axes.append(x)
        key = axis.replace("-raw", "").replace("-", "_")
        features[f"{key}_std"] = float(np.std(x))
        features[f"{key}_range"] = float(np.percentile(x, 95) - np.percentile(x, 5))
        features.update(_spectrum_features(x, fs, key, 0.08, 0.7))
    if len(axes) == 3:
        n = min(len(x) for x in axes)
        magnitude = np.sqrt(sum(a[:n] ** 2 for a in axes))
        features["acc_magnitude_std"] = float(np.std(magnitude))
        features.update(_spectrum_features(magnitude, fs, "acc_magnitude", 0.08, 0.7))

    ir, red = _signal(row.get("ir-raw")), _signal(row.get("red-raw"))
    if ir is not None and red is not None:
        n = min(len(ir), len(red))
        ir, red = ir[:n], red[:n]
        ir_ac = _bandpass(ir, fs, 0.5, min(4.0, fs * 0.45))
        red_ac = _bandpass(red, fs, 0.5, min(4.0, fs * 0.45))
        if ir_ac is not None and red_ac is not None:
            ir_dc, red_dc = abs(float(np.mean(ir))), abs(float(np.mean(red)))
            ir_mod = float(np.sqrt(np.mean(ir_ac**2))) / max(ir_dc, 1e-9)
            red_mod = float(np.sqrt(np.mean(red_ac**2))) / max(red_dc, 1e-9)
            features["red_ir_modulation_ratio"] = red_mod / max(ir_mod, 1e-9)
            features["ir_ac_dc"] = ir_mod
            features["red_ac_dc"] = red_mod

    features["ir_quality"] = _number(row.get("ir-quality"))
    features["red_quality"] = _number(row.get("red-quality"))

    rr_peak = math.nan
    rr_signal = _signal(row.get("ir-filtered-rr"))
    if rr_signal is not None:
        rr_features = _spectrum_features(rr_signal, fs, "direct_ir_rr", 0.08, 0.65)
        rr_peak = rr_features.get("direct_ir_rr_peak_rate", math.nan)
    return features, rr_peak


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t"}
    return False


def extract_features(archive_path: Path) -> pd.DataFrame:
    if CACHE.is_file():
        cached = pd.read_csv(CACHE)
        if not cached.empty and cached.iloc[0].get("cache_version") == CACHE_VERSION:
            data = cached.drop(columns=["cache_version"])
            if CACHE_META.is_file():
                data.attrs["archive_extraction"] = json.loads(CACHE_META.read_text())
            return data
    rows: list[dict[str, Any]] = []
    files_loaded = 0
    files_skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith("_processed.pkl")]
        print(f"Found {len(names)} per-person ring tables in the HealthRing archive.", flush=True)
        for file_index, name in enumerate(names, start=1):
            match = re.search(r"(\d{5})_ring([12])_processed\.pkl$", name, flags=re.I)
            if not match:
                continue
            participant, ring = match.group(1), f"ring{match.group(2)}"
            if archive.getinfo(name).file_size == 0:
                files_skipped += 1
                print(f"Skipping an empty recovered table ({file_index}/{len(names)}).", flush=True)
                continue
            try:
                with archive.open(name) as fh:
                    table = pd.read_pickle(fh)
            except (EOFError, pickle.UnpicklingError, ValueError, OSError) as exc:
                files_skipped += 1
                print(f"Skipping an unreadable recovered table ({file_index}/{len(names)}): {type(exc).__name__}.", flush=True)
                continue
            if not isinstance(table, pd.DataFrame):
                files_skipped += 1
                continue
            files_loaded += 1
            for _, record in table.iterrows():
                treadmill = _truthy(record.get("treadmill")) or "treadmill" in str(record.get("session_name", "")).lower()
                experiment = str(record.get("Experiment", "")).lower()
                if not treadmill and experiment in {"sport", "treadmill"}:
                    treadmill = True
                features, rr_peak = _features(record)
                if not features:
                    continue
                row: dict[str, Any] = {
                    "participant": participant,
                    "ring": ring,
                    "cohort": "treadmill" if treadmill else "controlled_daily",
                    "rr": _number(record.get("resp_rr")),
                    "spo2": _number(record.get("spo2")),
                    "hr": _number(record.get("hr")),
                    "direct_rr_peak": rr_peak,
                }
                row.update(features)
                rows.append(row)
            if file_index % 5 == 0 or file_index == len(names):
                print(f"Extracted ring-only features from {file_index}/{len(names)} tables ({len(rows)} windows).", flush=True)
    if not rows:
        raise RuntimeError("No HealthRing rows were read. Check that the HealthRing v2.1 archive is complete and contains RingDatasetV2/*.pkl.")
    data = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache_data = data.copy()
    cache_data.insert(0, "cache_version", CACHE_VERSION)
    cache_data.to_csv(CACHE, index=False)
    extraction = {
        "archive_filename": archive_path.name,
        "recovered_table_files_found": int(len(names)),
        "table_files_loaded": int(files_loaded),
        "empty_or_unreadable_table_files_skipped": int(files_skipped),
        "source_integrity_note": (
            "The Zenodo file checksum matches its record, but the source archive is missing its ZIP directory. "
            "This evaluation uses only the per-person tables recoverable by local ZIP repair and is therefore partial."
            if "repaired" in archive_path.name.lower()
            else "Source ZIP passed the standard ZIP directory check."
        ),
    }
    CACHE_META.write_text(json.dumps(extraction, indent=2) + "\n")
    data.attrs["archive_extraction"] = extraction
    return data


def _model_candidates() -> dict[str, Any]:
    prep = lambda model: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)
    return {
        "training_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees": prep(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4, max_features=0.8, n_jobs=-1, random_state=42)),
        "random_forest": prep(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.8, n_jobs=-1, random_state=42)),
        "hist_gradient_boosting": prep(HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7, min_samples_leaf=12, l2_regularization=5.0, random_state=42)),
    }


def _metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred, groups = y[ok], pred[ok], groups[ok]
    if not len(y):
        return {"n_windows": 0}
    subject_mae = [float(mean_absolute_error(y[groups == g], pred[groups == g])) for g in np.unique(groups)]
    rng = np.random.default_rng(20261127)
    if len(subject_mae) >= 2:
        values = np.asarray(subject_mae, dtype=np.float64)
        bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
        macro_mae_ci = [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]
    else:
        macro_mae_ci = None
    pearson = float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else None
    spearman = float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else None
    return {
        "n_windows": int(len(y)),
        "n_participants": int(len(np.unique(groups))),
        "mae": float(mean_absolute_error(y, pred)),
        "participant_macro_mae": float(np.mean(subject_mae)),
        "participant_macro_mae_95_ci": macro_mae_ci,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def _grouped_cv(frame: pd.DataFrame, target: str, *, ring: str, cohort: str) -> dict[str, Any]:
    subset = frame[(frame["ring"] == ring) & (frame["cohort"] == cohort)].copy()
    subset = subset[np.isfinite(pd.to_numeric(subset[target], errors="coerce"))]
    if target == "rr":
        subset = subset[subset[target].between(4, 40)]
    elif target == "spo2":
        subset = subset[subset[target].between(75, 100)]
    if subset["participant"].nunique() < 5 or len(subset) < 20:
        return {"status": "insufficient_data", "n_windows": int(len(subset)), "n_participants": int(subset["participant"].nunique())}
    excluded = {"participant", "ring", "cohort", "rr", "spo2", "hr", "direct_rr_peak"}
    feature_names = [c for c in subset.columns if c not in excluded and pd.api.types.is_numeric_dtype(subset[c])]
    X = subset[feature_names].to_numpy(dtype=np.float64)
    y = pd.to_numeric(subset[target], errors="coerce").to_numpy(dtype=np.float64)
    groups = subset["participant"].to_numpy()
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    result: dict[str, Any] = {"status": "evaluated", "task": target, "ring": ring, "cohort": cohort, "features": feature_names, "models": {}}
    direct = pd.to_numeric(subset["direct_rr_peak"], errors="coerce").to_numpy(dtype=np.float64)
    if target == "rr":
        result["models"]["direct_ring_ppg_spectral_peak"] = _metrics(y, direct, groups)
    for name, template in _model_candidates().items():
        oof = np.full(len(y), np.nan)
        fold_info = []
        for fold_id, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
            model = template
            model.fit(X[train], y[train])
            oof[test] = model.predict(X[test])
            fold_info.append({
                "fold": fold_id,
                "train_participants": int(len(np.unique(groups[train]))),
                "held_out_participants": int(len(np.unique(groups[test]))),
                "test_windows": int(len(test)),
            })
            if fold_id == 1:
                print(f"  {ring} {cohort} {target}: fitted {name}; evaluating {splitter.get_n_splits()} subject-held-out folds.", flush=True)
        result["models"][name] = {**_metrics(y, oof, groups), "folds": fold_info}
    result["n_windows"] = int(len(subset))
    result["n_participants"] = int(subset["participant"].nunique())
    result["target_range"] = {"min": float(np.min(y)), "max": float(np.max(y)), "median": float(np.median(y))}
    if target == "spo2":
        low = y < 90
        result["controlled_low_oxygen_windows"] = int(low.sum())
        result["controlled_low_oxygen_participants"] = int(len(np.unique(groups[low])))
        result["low_oxygen_task_note"] = "The <90% subset is controlled hypoxia in volunteers, not a pulmonary disease cohort."
    return result


def _nested_grouped_cv(frame: pd.DataFrame, target: str, *, ring: str, cohort: str) -> dict[str, Any]:
    """Choose a regression method inside each outer participant-held-out fold."""
    subset = frame[(frame["ring"] == ring) & (frame["cohort"] == cohort)].copy()
    subset = subset[np.isfinite(pd.to_numeric(subset[target], errors="coerce"))]
    if target == "rr":
        subset = subset[subset[target].between(4, 40)]
    elif target == "spo2":
        subset = subset[subset[target].between(75, 100)]
    groups = subset["participant"].to_numpy()
    if len(np.unique(groups)) < 8 or len(subset) < 40:
        return {"status": "insufficient_data", "n_windows": int(len(subset)), "n_participants": int(len(np.unique(groups)))}
    excluded = {"participant", "ring", "cohort", "rr", "spo2", "hr", "direct_rr_peak"}
    feature_names = [c for c in subset.columns if c not in excluded and pd.api.types.is_numeric_dtype(subset[c])]
    X = subset[feature_names].to_numpy(dtype=np.float64)
    y = pd.to_numeric(subset[target], errors="coerce").to_numpy(dtype=np.float64)
    outer = GroupKFold(n_splits=5)
    candidates = tuple(name for name in _model_candidates() if name != "training_mean")
    oof = np.full(len(y), np.nan, dtype=np.float64)
    selected: list[dict[str, Any]] = []
    for outer_id, (train, test) in enumerate(outer.split(X, y, groups), start=1):
        train_groups = groups[train]
        inner = GroupKFold(n_splits=min(3, len(np.unique(train_groups))))
        inner_scores: dict[str, float] = {}
        for name in candidates:
            inner_oof = np.full(len(train), np.nan, dtype=np.float64)
            for inner_train, inner_test in inner.split(X[train], y[train], train_groups):
                model = _model_candidates()[name]
                model.fit(X[train][inner_train], y[train][inner_train])
                inner_oof[inner_test] = model.predict(X[train][inner_test])
            inner_scores[name] = _metrics(y[train], inner_oof, train_groups)["participant_macro_mae"]
        chosen = min(inner_scores, key=inner_scores.get)
        model = _model_candidates()[chosen]
        model.fit(X[train], y[train])
        oof[test] = model.predict(X[test])
        selected.append({
            "outer_fold": outer_id,
            "selected_model_inside_training_participants": chosen,
            "inner_grouped_participant_macro_mae": {key: float(value) for key, value in inner_scores.items()},
            "held_out_participants": int(len(np.unique(groups[test]))),
            "held_out_windows": int(len(test)),
        })
        print(f"  nested {ring} {cohort} {target}: outer fold {outer_id}/5 selected {chosen} using inner participant folds.", flush=True)
    result = _metrics(y, oof, groups)
    result.update({
        "status": "evaluated",
        "task": target,
        "ring": ring,
        "cohort": cohort,
        "validation": "nested five-fold GroupKFold by participant; 3-fold inner GroupKFold selects among ridge, Extra Trees, random forest, and histogram gradient boosting; the outer participant fold is untouched until scoring",
        "n_windows": int(len(subset)),
        "n_participants": int(len(np.unique(groups))),
        "selected_models_by_outer_fold": selected,
    })
    return result


def _treadmill_transfer(frame: pd.DataFrame) -> dict[str, Any]:
    train = frame[(frame["ring"] == "ring1") & (frame["cohort"] == "controlled_daily") & frame["rr"].between(4, 40)].copy()
    test = frame[(frame["ring"] == "ring1") & (frame["cohort"] == "treadmill") & frame["rr"].between(4, 40)].copy()
    if train["participant"].nunique() < 5 or test["participant"].nunique() < 5:
        return {"status": "insufficient_data", "training_participants": int(train["participant"].nunique()), "test_participants": int(test["participant"].nunique())}
    excluded = {"participant", "ring", "cohort", "rr", "spo2", "hr", "direct_rr_peak"}
    feature_names = [c for c in train.columns if c not in excluded and pd.api.types.is_numeric_dtype(train[c])]
    X_train, X_test = train[feature_names].to_numpy(dtype=np.float64), test[feature_names].to_numpy(dtype=np.float64)
    y_train = train["rr"].to_numpy(dtype=np.float64)
    y_test = test["rr"].to_numpy(dtype=np.float64)
    groups = test["participant"].to_numpy()
    results: dict[str, Any] = {
        "status": "evaluated",
        "task": "zero-shot RR transfer from HealthRing controlled/daily Ring 1 to held-out treadmill Ring 1 participants",
        "training_participants": int(train["participant"].nunique()),
        "test_participants": int(test["participant"].nunique()),
        "test_windows": int(len(test)),
        "models": {},
    }
    results["models"]["direct_ring_ppg_spectral_peak"] = _metrics(y_test, test["direct_rr_peak"].to_numpy(dtype=np.float64), groups)
    for name in ("ridge", "extra_trees", "random_forest", "hist_gradient_boosting"):
        model = _model_candidates()[name]
        model.fit(X_train, y_train)
        results["models"][name] = _metrics(y_test, model.predict(X_test), groups)
    return results


def build_report(data: pd.DataFrame) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for ring in ("ring1", "ring2"):
        for target in ("rr", "spo2"):
            summaries[f"{ring}_{target}"] = _grouped_cv(data, target, ring=ring, cohort="controlled_daily")
    summaries["ring1_treadmill_rr"] = _grouped_cv(data, "rr", ring="ring1", cohort="treadmill")
    nested = {
        "ring1_rr": _nested_grouped_cv(data, "rr", ring="ring1", cohort="controlled_daily"),
        "ring2_rr": _nested_grouped_cv(data, "rr", ring="ring2", cohort="controlled_daily"),
    }
    return {
        "model_version": "plumoring-healthring-ring-vitals-0.1",
        "dataset": "HealthRing RingDatasetV2.1",
        "dataset_doi": "10.5281/zenodo.18426864",
        "license": "CC BY 4.0",
        "input_policy": "Ring-only inputs: red/infrared ring PPG, ring accelerometer and ring-derived transforms. No smartwatch, chest ECG, oximeter waveform, respiration band, reference HR/SpO2/RR, or clinical labels are predictors.",
        "targets": {
            "rr": "Respiratory-rate reference from synchronized respiratory instrumentation (or chest ECG-derived RR in the separate treadmill cohort).",
            "spo2": "Synchronized reference SpO2; this is measurement estimation, not a disease diagnosis.",
        },
        "split": "Five-fold GroupKFold by participant; all windows from a participant remain in one held-out fold. Ring types are evaluated separately.",
        "counts": {
            "windows": int(len(data)),
            "participants": int(data["participant"].nunique()),
            "rings": {key: int(value) for key, value in data.groupby("ring")["participant"].nunique().items()},
            "cohorts": {key: {"windows": int(len(group)), "participants": int(group["participant"].nunique())} for key, group in data.groupby("cohort")},
            "ring_quality_nonmissing": {key: float(pd.to_numeric(data[key], errors="coerce").notna().mean()) for key in ("ir_quality", "red_quality") if key in data},
        },
        "archive_extraction": data.attrs.get("archive_extraction", {}),
        "grouped_subject_held_out_results": summaries,
        "nested_method_selection_results": nested,
        "zero_shot_treadmill_transfer": _treadmill_transfer(data),
        "deployment_status": "research_only_no_disease_classifier",
        "clinical_limitations": [
            "HealthRing contains healthy volunteers, not participants with COPD, asthma, pneumonia, or other confirmed pulmonary diagnoses.",
            "Controlled low oxygen in volunteers is not equivalent to a pulmonary disease or a spontaneous exacerbation.",
            "The dataset validates ring signal-to-vital estimation only; it cannot train or validate disease detection.",
            "Performance on this custom research ring does not establish performance on an Oura, Samsung, or future PlumoRing device.",
            "Reference measurements and participant-level uncertainty must be reviewed before any physiological estimate is shown as more than a research demonstration.",
        ],
        "reproduction": "python scripts/evaluate_healthring_ring_vitals.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="HealthRing v2.1 Zenodo archive")
    parser.add_argument("--output", type=Path, default=REPORT, help="Aggregate metrics JSON")
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"HealthRing archive not found: {args.archive}")
    if not zipfile.is_zipfile(args.archive):
        recovered = args.archive.with_name("RingDatasetV2.1_repaired.zip")
        if zipfile.is_zipfile(recovered):
            print(f"The source ZIP is missing its index; using locally recovered entries from {recovered.name}.", flush=True)
            args.archive = recovered
        else:
            raise SystemExit("HealthRing archive lacks a readable ZIP directory and no recovered archive is available.")
    data = extract_features(args.archive)
    print("Comparing physics-based and machine-learning estimators with participant-held-out validation...", flush=True)
    report = build_report(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {"counts": report["counts"], "results": report["grouped_subject_held_out_results"], "transfer": report["zero_shot_treadmill_transfer"]}
    print(json.dumps(compact, indent=2))
    print(f"\nWrote aggregate-only report: {args.output}")


if __name__ == "__main__":
    main()
