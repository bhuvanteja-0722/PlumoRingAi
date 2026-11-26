#!/usr/bin/env python3
"""Participant-held-out feasibility study of asthma worsening from wearable data.

The target is a weekly self-report of systemic corticosteroid use above the
participant's usual amount. Only smartwatch heart-rate and activity readings
from that reported week are used. This is retrospective week-level
classification, not prospective prediction, continuous event detection, or a
diagnosis of asthma in an undiagnosed person.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "aamos00" / "raw"
REPORT_PATH = ROOT / "models" / "aamos_asthma_worsening_metrics.json"
WEEK_DAYS = 7
MIN_WEAR_DAYS = 3
WATCH_FILES = tuple(DATA_DIR / f"anonym_aamos00_smartwatch{i}.csv" for i in (1, 2, 3))
WEEKLY_FILE = DATA_DIR / "anonym_aamos00_weeklyquestionnaire.csv"


def _watch_daily_features() -> pd.DataFrame:
    missing = [path.name for path in (*WATCH_FILES, WEEKLY_FILE) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing AAMOS-00 files in {DATA_DIR}: {', '.join(missing)}")
    watch = pd.concat((pd.read_csv(path) for path in WATCH_FILES), ignore_index=True)
    for column in ("user_key", "date", "activity_type", "intensity", "steps", "hr"):
        if column not in watch:
            raise ValueError(f"AAMOS smartwatch data is missing {column!r}")
    watch["user_key"] = watch["user_key"].astype(str)
    watch["date"] = pd.to_numeric(watch["date"], errors="coerce")
    for column in ("activity_type", "intensity", "steps", "hr"):
        watch[column] = pd.to_numeric(watch[column], errors="coerce")
    watch = watch.dropna(subset=["user_key", "date"])
    watch["hr"] = watch["hr"].where(watch["hr"].between(25, 220))
    watch["steps"] = watch["steps"].where(watch["steps"].between(0, 250))
    watch["intensity"] = watch["intensity"].where(watch["intensity"].between(0, 255))
    # Activity codes follow the study's data dictionary. They are device
    # classifications, not symptoms or diagnosis labels.
    activity = watch["activity_type"]
    watch["is_walk"] = activity.isin([1, 16, 33]).astype(float)
    watch["is_run"] = activity.isin([66, 82, 98]).astype(float)
    watch["is_sleep"] = activity.isin([106, 112, 121, 122, 123]).astype(float)
    watch["is_sedentary"] = activity.isin([80, 89, 90, 91, 92, 96]).astype(float)
    watch["is_worn"] = ~activity.isin([3, 6])
    daily = watch.groupby(["user_key", "date"], sort=False).agg(
        hr_mean=("hr", "mean"),
        hr_std=("hr", "std"),
        hr_p10=("hr", lambda x: x.quantile(0.10)),
        hr_p90=("hr", lambda x: x.quantile(0.90)),
        hr_valid_minutes=("hr", "count"),
        steps=("steps", "sum"),
        intensity_mean=("intensity", "mean"),
        intensity_p90=("intensity", lambda x: x.quantile(0.90)),
        walk_minutes=("is_walk", "sum"),
        run_minutes=("is_run", "sum"),
        sleep_minutes=("is_sleep", "sum"),
        sedentary_minutes=("is_sedentary", "sum"),
        worn_minutes=("is_worn", "sum"),
    ).reset_index()
    daily["steps"] = np.log1p(daily["steps"].clip(lower=0))
    return daily


def _target_is_positive(row: pd.Series) -> bool | None:
    # The official data dictionary: 1=no; 2=yes, no more than usual;
    # 3=yes, more than usual; 4=unsure. The original AAMOS example script
    # also defines an OCS event specifically as weekly_oral == 3.
    value = pd.to_numeric(pd.Series([row.get("weekly_oral")]), errors="coerce").iloc[0]
    if pd.isna(value) or value == 4:
        return None
    if value not in (1, 2, 3):
        return None
    return bool(value == 3)


def _summarize_window(window: pd.DataFrame, participant: str, survey_day: int) -> dict[str, float]:
    window = window.sort_values("date", kind="stable")
    features: dict[str, float] = {}
    columns = (
        "hr_mean", "hr_std", "hr_p10", "hr_p90", "hr_valid_minutes",
        "steps", "intensity_mean", "intensity_p90", "walk_minutes",
        "run_minutes", "sleep_minutes", "sedentary_minutes", "worn_minutes",
    )
    for name in columns:
        values = pd.to_numeric(window[name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        values = values[valid]
        if values.size == 0:
            features[f"{name}_mean"] = np.nan
            features[f"{name}_std"] = np.nan
            features[f"{name}_max"] = np.nan
            features[f"{name}_slope"] = np.nan
        else:
            features[f"{name}_mean"] = float(np.mean(values))
            features[f"{name}_std"] = float(np.std(values))
            features[f"{name}_max"] = float(np.max(values))
            positions = window["date"].to_numpy(dtype=float)[valid]
            features[f"{name}_slope"] = float(np.polyfit(positions, values, 1)[0]) if values.size > 1 else 0.0
    features["watch_days_with_data"] = float(window["date"].nunique())
    features["valid_hr_minutes_total"] = float(window["hr_valid_minutes"].fillna(0).sum())
    return features


def _build_examples(daily: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    weekly = pd.read_csv(WEEKLY_FILE)
    required = {"user_key", "date", "weekly_oral"}
    if not required.issubset(weekly.columns):
        raise ValueError(f"Weekly questionnaire is missing {sorted(required - set(weekly.columns))}")
    weekly["user_key"] = weekly["user_key"].astype(str)
    weekly["date"] = pd.to_numeric(weekly["date"], errors="coerce")
    daily_by_user = {user: frame.set_index("date", drop=False) for user, frame in daily.groupby("user_key", sort=False)}
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[str] = []
    excluded_unknown = 0
    excluded_coverage = 0
    eligible_positive_users: set[str] = set()
    for _, survey in weekly.iterrows():
        label = _target_is_positive(survey)
        if label is None:
            excluded_unknown += 1
            continue
        participant = str(survey["user_key"])
        if participant not in daily_by_user or pd.isna(survey["date"]):
            excluded_coverage += 1
            continue
        day = int(survey["date"])
        watch = daily_by_user[participant]
        expected_days = list(range(day - WEEK_DAYS + 1, day + 1))
        window = watch.loc[watch.index.intersection(expected_days)].reset_index(drop=True)
        if window["date"].nunique() < MIN_WEAR_DAYS:
            excluded_coverage += 1
            continue
        feature_row = _summarize_window(window, participant, day)
        rows.append(feature_row)
        labels.append(int(label))
        groups.append(participant)
        if label:
            eligible_positive_users.add(participant)
    if not rows or len(set(labels)) != 2:
        raise RuntimeError("AAMOS watch/questionnaire join did not yield both classes; check date alignment and labels.")
    table = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = np.asarray(labels, dtype=np.int8)
    group_array = np.asarray(groups)
    counts = {
        "weekly_questionnaire_rows": int(len(weekly)),
        "eligible_labeled_weeks_with_at_least_3_watch_days": int(len(y)),
        "positive_weeks_reported_above_usual_ocs": int(y.sum()),
        "negative_weeks_no_increase_in_ocs": int((y == 0).sum()),
        "eligible_participants": int(len(np.unique(group_array))),
        "participants_with_positive_weeks": int(len(eligible_positive_users)),
        "excluded_unknown_or_missing_ocs_response": int(excluded_unknown),
        "excluded_missing_or_insufficient_watch_coverage": int(excluded_coverage),
        "watch_days_required_in_week": MIN_WEAR_DAYS,
        "watch_week_window": "survey date minus six days through survey date, inclusive; label reports past-week OCS use",
        "watch_day_rows": int(len(daily)),
        "watch_participants": int(daily["user_key"].nunique()),
        "watch_day_rows_with_valid_hr": int((daily["hr_valid_minutes"] > 0).sum()),
    }
    return table.to_numpy(dtype=float), y, group_array, list(table.columns), counts


def _models() -> dict[str, Any]:
    return {
        "participant_prevalence_baseline": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            DummyClassifier(strategy="prior"),
        ),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=2026),
        ),
        "rbf_support_vector_machine": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            SVC(C=0.5, gamma="scale", class_weight="balanced", probability=True, random_state=2026),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=5, l2_regularization=8, random_state=2026),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.7, class_weight="balanced_subsample", n_jobs=-1, random_state=2026),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.7, class_weight="balanced", n_jobs=-1, random_state=2026),
        ),
    }


def _metrics(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    predicted = scores >= 0.5
    return {
        "roc_auc_equal_participant_weight": float(roc_auc_score(y, scores, sample_weight=weights)) if len(np.unique(y)) == 2 else None,
        "average_precision_equal_participant_weight": float(average_precision_score(y, scores, sample_weight=weights)),
        "average_precision_pooled_weeks": float(average_precision_score(y, scores)),
        "threshold_0_5_sensitivity_equal_participant_weight": float(np.average(predicted[y == 1], weights=weights[y == 1])) if np.any(y == 1) else None,
        "threshold_0_5_specificity_equal_participant_weight": float(np.average(~predicted[y == 0], weights=weights[y == 0])) if np.any(y == 0) else None,
        "threshold_0_5_balanced_accuracy_pooled_weeks": float(balanced_accuracy_score(y, predicted)),
    }


def _fit_model(model: Any, name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
    group_sizes = pd.Series(groups).value_counts().to_dict()
    sample_weights = np.asarray([1.0 / group_sizes[group] for group in groups], dtype=float)
    sample_weights *= len(sample_weights) / sample_weights.sum()
    if name == "participant_prevalence_baseline":
        model.fit(X, y)
        return
    if name == "hist_gradient_boosting":
        sample_weights *= compute_sample_weight("balanced", y)
    model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weights})


def _grouped_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    models = _models()
    predictions = {name: np.full(len(y), np.nan, dtype=float) for name in models}
    folds_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=2026)
    for fold, (train, heldout) in enumerate(splitter.split(X, y, groups), start=1):
        if len(np.unique(y[train])) < 2:
            continue
        for name, model in models.items():
            _fit_model(model, name, X[train], y[train], groups[train])
            scores = model.predict_proba(X[heldout])[:, 1]
            predictions[name][heldout] = scores
            fold_weights_map = pd.Series(groups[heldout]).value_counts().to_dict()
            fold_weights = np.asarray([1.0 / fold_weights_map[group] for group in groups[heldout]], dtype=float)
            fold_weights *= len(fold_weights) / fold_weights.sum()
            folds_by_model[name].append({
                "fold": fold,
                "held_out_participants": int(len(np.unique(groups[heldout]))),
                "held_out_positive_participants": int(len(set(groups[heldout][y[heldout] == 1]))),
                "weeks": int(len(heldout)),
                "positive_weeks": int(y[heldout].sum()),
                "average_precision_equal_participant_weight": float(average_precision_score(y[heldout], scores, sample_weight=fold_weights)) if np.any(y[heldout] == 1) else None,
                "roc_auc_equal_participant_weight": float(roc_auc_score(y[heldout], scores, sample_weight=fold_weights)) if len(np.unique(y[heldout])) == 2 else None,
            })
    return predictions, folds_by_model


def _nested_selection(X: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_splits: int) -> tuple[dict[str, Any], dict[str, int]]:
    selected_predictions = np.full(len(y), np.nan, dtype=float)
    selected_threshold_predictions = np.full(len(y), False, dtype=bool)
    selected_models: dict[str, int] = {}
    fold_reports: list[dict[str, Any]] = []
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=2027)
    for outer_fold, (train, heldout) in enumerate(outer.split(X, y, groups), start=1):
        train_groups = groups[train]
        train_positive_people = len(set(train_groups[y[train] == 1]))
        inner_splits = min(3, len(np.unique(train_groups)), train_positive_people)
        if inner_splits < 2 or len(np.unique(y[train])) < 2:
            continue
        inner_predictions, _ = _grouped_oof(X[train], y[train], train_groups, inner_splits)
        train_sizes = pd.Series(train_groups).value_counts().to_dict()
        inner_weights = np.asarray([1.0 / train_sizes[group] for group in train_groups], dtype=float)
        inner_weights *= len(inner_weights) / inner_weights.sum()
        inner_scores: dict[str, tuple[float, float]] = {}
        for name, scores in inner_predictions.items():
            valid = np.isfinite(scores)
            if not np.any(valid) or not np.any(y[train][valid] == 1):
                continue
            ap = float(average_precision_score(y[train][valid], scores[valid], sample_weight=inner_weights[valid]))
            auc = float(roc_auc_score(y[train][valid], scores[valid], sample_weight=inner_weights[valid])) if len(np.unique(y[train][valid])) == 2 else 0.0
            inner_scores[name] = (ap, auc)
        if not inner_scores:
            chosen = "logistic_regression"
        else:
            chosen = max(inner_scores, key=lambda name: (inner_scores[name][0], inner_scores[name][1]))
        inner_selected = inner_predictions.get(chosen, np.full(len(train), np.nan))
        negative = np.isfinite(inner_selected) & (y[train] == 0)
        if np.any(negative):
            negative_scores = inner_selected[negative]
            negative_groups = train_groups[negative]
            negative_group_sizes = pd.Series(negative_groups).value_counts().to_dict()
            negative_weights = np.asarray([1.0 / negative_group_sizes[group] for group in negative_groups], dtype=float)
            negative_weights *= len(negative_weights) / negative_weights.sum()
            order = np.argsort(negative_scores)
            cumulative = np.cumsum(negative_weights[order])
            threshold = float(negative_scores[order[min(np.searchsorted(cumulative, 0.90 * cumulative[-1]), len(order) - 1)]])
        else:
            threshold = 0.5
        model = _models()[chosen]
        _fit_model(model, chosen, X[train], y[train], train_groups)
        scores = model.predict_proba(X[heldout])[:, 1]
        selected_predictions[heldout] = scores
        selected_threshold_predictions[heldout] = scores >= threshold
        selected_models[chosen] = selected_models.get(chosen, 0) + 1
        held_sizes = pd.Series(groups[heldout]).value_counts().to_dict()
        held_weights = np.asarray([1.0 / held_sizes[group] for group in groups[heldout]], dtype=float)
        held_weights *= len(held_weights) / held_weights.sum()
        fold_reports.append({
            "outer_fold": outer_fold,
            "selected_model_using_inner_grouped_cv": chosen,
            "held_out_participants": int(len(np.unique(groups[heldout]))),
            "held_out_positive_participants": int(len(set(groups[heldout][y[heldout] == 1]))),
            "weeks": int(len(heldout)),
            "positive_weeks": int(y[heldout].sum()),
            "threshold_chosen_in_inner_training_data_for_90pct_specificity": threshold,
            "average_precision_equal_participant_weight": float(average_precision_score(y[heldout], scores, sample_weight=held_weights)) if np.any(y[heldout] == 1) else None,
            "roc_auc_equal_participant_weight": float(roc_auc_score(y[heldout], scores, sample_weight=held_weights)) if len(np.unique(y[heldout])) == 2 else None,
        })
    valid = np.isfinite(selected_predictions)
    sizes = pd.Series(groups[valid]).value_counts().to_dict()
    weights = np.asarray([1.0 / sizes[group] for group in groups[valid]], dtype=float)
    weights *= len(weights) / weights.sum()
    truth = y[valid]
    predictions = selected_threshold_predictions[valid]
    sensitivity = float(np.average(predictions[truth == 1], weights=weights[truth == 1])) if np.any(truth == 1) else None
    specificity = float(np.average(~predictions[truth == 0], weights=weights[truth == 0])) if np.any(truth == 0) else None
    ppv = float(np.average(truth[predictions], weights=weights[predictions])) if np.any(predictions) else 0.0
    return {
        "selection_method": "nested participant-held-out CV: 5 outer folds; model chosen in 3-fold participant CV within each outer training set",
        "selected_models_by_outer_fold": selected_models,
        "metrics": _metrics(y[valid], selected_predictions[valid], weights),
        "operating_point": {
            "threshold_selection": "weighted 90th percentile of negative inner-CV scores, refit and applied to the outer held-out participants",
            "sensitivity_equal_participant_weight": sensitivity,
            "specificity_equal_participant_weight": specificity,
            "positive_predictive_value_equal_participant_weight": ppv,
            "false_positive_rate_equal_participant_weight": float(np.average(predictions[truth == 0], weights=weights[truth == 0])) if np.any(truth == 0) else None,
        },
        "folds": fold_reports,
    }, selected_models


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_names: list[str], counts: dict[str, Any]) -> dict[str, Any]:
    n_people = len(np.unique(groups))
    positive_people = len(set(groups[y == 1]))
    if n_people < 5 or positive_people < 2:
        raise RuntimeError("Need at least five participants and two positive participants for grouped evaluation.")
    fold_count = min(5, n_people)
    group_sizes = pd.Series(groups).value_counts().to_dict()
    weights = np.asarray([1.0 / group_sizes[group] for group in groups], dtype=float)
    weights *= len(weights) / weights.sum()
    reports: dict[str, Any] = {}
    oof_predictions, folds_by_model = _grouped_oof(X, y, groups, fold_count)
    for name, oof in oof_predictions.items():
        valid = np.isfinite(oof)
        reports[name] = {
            "metrics": _metrics(y[valid], oof[valid], weights[valid]),
            "folds": folds_by_model[name],
        }
    prevalence = float(np.average(y, weights=weights))
    nested_report, selected_models = _nested_selection(X, y, groups, fold_count)
    best_name = max(
        selected_models,
        key=lambda name: selected_models[name],
    ) if selected_models else max(
        (name for name in reports if name != "participant_prevalence_baseline"),
        key=lambda name: (
            reports[name]["metrics"]["average_precision_equal_participant_weight"] or 0,
            reports[name]["metrics"]["roc_auc_equal_participant_weight"] or 0,
        ),
    )
    best = nested_report["metrics"]
    auc = float(best["roc_auc_equal_participant_weight"] or 0)
    ap = float(best["average_precision_equal_participant_weight"] or 0)
    sens_at_controlled_specificity = float(nested_report["operating_point"]["sensitivity_equal_participant_weight"] or 0)
    passed = ap >= prevalence * 1.25 and auc >= 0.60 and sens_at_controlled_specificity >= 0.50
    return {
        "model_version": "plumoring-aamos-asthma-worsening-feasibility-0.1",
        "dataset": "AAMOS-00 · asthma cohort with MiBand3 smartwatch and weekly questionnaire",
        "dataset_doi": "10.7488/ds/3775",
        "dataset_url": "https://datashare.ed.ac.uk/items/8478e384-fd1b-4a37-9555-0c6e1218e90b",
        "task": "retrospective classification of a past-week report of systemic corticosteroid use above usual in adults with known asthma",
        "outcome_definition": "weekly_oral == 3: yes, more systemic corticosteroids than usual; codes 1 and 2 are negative for increased use; unsure/missing responses are excluded",
        "input_channels": ["smartwatch heart-rate summaries", "smartwatch step/activity summaries"],
        "excluded_inputs": ["symptom questionnaires", "weekly outcome fields", "inhaler usage", "peak flow", "environmental fields", "future readings"],
        "limitations": [
            "This is a retrospective week-level classification because the questionnaire asks about the past week; it is not a future attack forecast.",
            "The label is a self-reported increase in systemic corticosteroid use, not an independently adjudicated or timestamped attack diagnosis.",
            "The cohort is small (22 study participants overall, with fewer contributing both watch data and labeled weeks).",
            "Inputs are MiBand3 smartwatch heart rate and activity, not raw ring PPG, ring SpO2, or PlumoRing sensor data.",
            "This evaluates worsening in adults already known to have asthma; it cannot diagnose asthma in an undiagnosed person.",
            "No sensor stream in this dataset supports pneumonia, infection, bronchitis, or COPD diagnosis.",
            "Results are research-only and require a larger external cohort and clinical review before any care use.",
        ],
        "features": feature_names,
        "split": f"{fold_count}-fold StratifiedGroupKFold by participant; each participant is held out from all model fitting in their fold",
        "counts": counts,
        "equal_participant_weight_positive_prevalence": prevalence,
        "models": reports,
        "best_candidate": f"Nested selection (most frequent: {best_name})",
        "best_candidate_metrics": best,
        "nested_model_selection": nested_report,
        "exploratory_gate": {
            "criteria": "nested equal-participant average precision >= 1.25x participant-weighted prevalence; ROC-AUC >= 0.60; sensitivity >= 0.50 at a 90% specificity target selected inside training folds",
            "passed": passed,
            "patient_facing_or_diagnostic": False,
        },
        "deployment_status": "research_only_not_deployed",
        "reproduction": "python scripts/evaluate_aamos_asthma_worsening.py",
        "license": "CC BY 4.0; cite the dataset paper and dataset DOI before reuse",
    }


def main() -> None:
    global DATA_DIR, WATCH_FILES, WEEKLY_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    WATCH_FILES = tuple(DATA_DIR / f"anonym_aamos00_smartwatch{i}.csv" for i in (1, 2, 3))
    WEEKLY_FILE = DATA_DIR / "anonym_aamos00_weeklyquestionnaire.csv"
    print("Summarizing AAMOS smartwatch readings and joining weekly asthma labels...", flush=True)
    daily = _watch_daily_features()
    X, y, groups, names, counts = _build_examples(daily)
    report = evaluate(X, y, groups, names, counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "counts": report["counts"],
        "best_candidate": report["best_candidate"],
        "best_candidate_metrics": report["best_candidate_metrics"],
        "nested_operating_point": report["nested_model_selection"]["operating_point"],
        "no_skill_prevalence": report["equal_participant_weight_positive_prevalence"],
        "gate": report["exploratory_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
