#!/usr/bin/env python3
"""Train and evaluate a small causal GRU research model on PhysioNet Challenge 2019."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import joblib
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
    "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1",
    "Unit2", "HospAdmTime", "ICULOS",
)
FEATURE_KEYS = FEATURES
BOUNDS = {
    "HR": (25, 240), "O2Sat": (50, 100), "Temp": (25, 45), "SBP": (30, 300), "MAP": (20, 220),
    "DBP": (10, 200), "Resp": (3, 65), "EtCO2": (0, 120), "BaseExcess": (-40, 40), "HCO3": (2, 80),
    "FiO2": (.15, 100), "pH": (6.5, 8.2), "PaCO2": (5, 180), "SaO2": (40, 100), "AST": (0, 20000),
    "BUN": (0, 500), "Alkalinephos": (0, 20000), "Calcium": (1, 25), "Chloride": (40, 180),
    "Creatinine": (0, 60), "Bilirubin_direct": (0, 100), "Glucose": (5, 3000), "Lactate": (0, 60),
    "Magnesium": (0, 30), "Phosphate": (0, 60), "Potassium": (1, 20), "Bilirubin_total": (0, 150),
    "TroponinI": (0, 100000), "Hct": (5, 85), "Hgb": (1, 40), "PTT": (5, 500), "WBC": (0, 1000),
    "Fibrinogen": (0, 10000), "Platelets": (1, 5000), "Age": (0, 120), "Gender": (0, 1),
    "Unit1": (0, 1), "Unit2": (0, 1), "HospAdmTime": (-300000, 300000), "ICULOS": (1, 2000),
}
DEFAULTS = {
    "HR": 85, "O2Sat": 97, "Temp": 36.8, "SBP": 120, "MAP": 80, "DBP": 65, "Resp": 18,
    "EtCO2": 35, "BaseExcess": 0, "HCO3": 24, "FiO2": .21, "pH": 7.4, "PaCO2": 40, "SaO2": 97,
    "AST": 25, "BUN": 15, "Alkalinephos": 80, "Calcium": 9, "Chloride": 102, "Creatinine": 1,
    "Bilirubin_direct": .2, "Glucose": 100, "Lactate": 1.2, "Magnesium": 2, "Phosphate": 3.5,
    "Potassium": 4, "Bilirubin_total": .7, "TroponinI": 0, "Hct": 40, "Hgb": 13, "PTT": 30,
    "WBC": 8, "Fibrinogen": 300, "Platelets": 250, "Age": 65, "Gender": 0, "Unit1": 0, "Unit2": 0,
    "HospAdmTime": -24, "ICULOS": 1,
}
LABEL_KEY = "SepsisLabel"
SEED = 2026


@dataclass
class Record:
    values: np.ndarray
    observed: np.ndarray
    labels: np.ndarray
    source: str


class CausalSepsisNet(nn.Module):
    """Compact multi-scale temporal conv + BiGRU-attention classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(len(FEATURES) * 2, 48, 3, padding=1), nn.BatchNorm1d(48), nn.GELU())
        self.branch3 = nn.Conv1d(48, 24, 3, padding=1)
        self.branch5 = nn.Conv1d(48, 24, 5, padding=2)
        self.mix = nn.Sequential(nn.Conv1d(48, 48, 1), nn.BatchNorm1d(48))
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(48, 12), nn.GELU(), nn.Linear(12, 48), nn.Sigmoid())
        self.norm = nn.LayerNorm(48)
        self.gru = nn.GRU(48, 48, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(96, 3, batch_first=True, dropout=.1)
        self.attention_norm = nn.LayerNorm(96)
        self.head = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.LayerNorm(96), nn.Dropout(.25), nn.Linear(96, 1))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.transpose(1, 2)
        stem = self.stem(x)
        residual = self.mix(torch.cat((torch.nn.functional.gelu(self.branch3(stem)), torch.nn.functional.gelu(self.branch5(stem))), dim=1))
        gate = self.se(residual).unsqueeze(-1)
        x = self.norm((stem + residual * gate).transpose(1, 2))
        x, _ = self.gru(x)
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + attended)
        pooled = torch.cat((x.mean(dim=1), x.max(dim=1).values), dim=-1)
        return self.head(pooled).squeeze(-1)


def read_record(path: Path, source: str) -> Record | None:
    try:
        with path.open(newline="") as file:
            reader = csv.DictReader(file, delimiter="|")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if not rows or any(name not in (reader.fieldnames or []) for name in (*FEATURES, LABEL_KEY)):
        return None
    values = np.full((len(rows), len(FEATURES)), np.nan, dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(FEATURES):
            try:
                value = float(row[name])
                low, high = BOUNDS[name]
                if np.isfinite(value) and low <= value <= high:
                    values[i, j] = value
            except (ValueError, TypeError):
                pass
        try:
            labels[i] = float(row[LABEL_KEY]) == 1.0
        except (ValueError, TypeError):
            pass
    return Record(values, np.isfinite(values), labels, source)


def records_in(folder: Path, source: str) -> list[Record]:
    paths = sorted(folder.glob("*.psv"))
    records: list[Record] = []
    for index, path in enumerate(paths, 1):
        record = read_record(path, source)
        if record is not None and len(record.labels) >= 8:
            records.append(record)
        if index % 5000 == 0:
            print(f"Read {index:,}/{len(paths):,} records from {folder.name}.", flush=True)
    return records


def fit_scaling(records: Iterable[Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = np.concatenate([r.values for r in records], axis=0)
    defaults = np.array([DEFAULTS[name] for name in FEATURES], np.float32)
    medians, means, stds = defaults.copy(), defaults.copy(), np.ones(len(FEATURES), np.float32)
    for column in range(len(FEATURES)):
        observed = merged[:, column]
        observed = observed[np.isfinite(observed)]
        if len(observed):
            medians[column] = np.median(observed)
            means[column] = np.mean(observed)
            deviation = np.std(observed)
            stds[column] = deviation if deviation > 1e-4 else 1.0
    return medians, means, stds


def build_features(record: Record, medians: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    filled = record.values.copy()
    last = medians.copy()
    for t in range(len(filled)):
        mask = record.observed[t]
        last[mask] = filled[t, mask]
        filled[t, ~mask] = last[~mask]
    standardized = np.clip((filled - means) / stds, -5, 5)
    return np.concatenate((standardized, record.observed.astype(np.float32)), axis=1).astype(np.float32)


def choose_coordinates(records: list[Record], limit: int, rng: np.random.Generator, balanced: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.array([len(r.labels) for r in records], dtype=np.int64)
    y = np.concatenate([r.labels for r in records]).astype(np.int8)
    total = len(y)
    if total <= limit:
        flat = np.arange(total, dtype=np.int64)
    else:
        # Guarantee each patient contributes at least one window before sampling extra hours.
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        patient_count = min(len(records), limit)
        included_records = np.arange(len(records)) if patient_count == len(records) else rng.choice(len(records), patient_count, replace=False)
        first_rows = starts[included_records] + np.array([rng.integers(lengths[i]) for i in included_records])
        available = np.ones(total, dtype=bool)
        available[first_rows] = False
        flat_parts = [first_rows]
        slots = limit - len(first_rows)
        if balanced and slots:
            target_positive = max(int(limit * .125), int(y[first_rows].sum()))
            positive_candidates = np.flatnonzero((y == 1) & available)
            extra_positive = min(max(0, target_positive - int(y[first_rows].sum())), len(positive_candidates), slots)
            if extra_positive:
                selected_positive = rng.choice(positive_candidates, extra_positive, replace=False)
                flat_parts.append(selected_positive)
                available[selected_positive] = False
                slots -= extra_positive
        if slots:
            candidates = np.flatnonzero(available)
            flat_parts.append(rng.choice(candidates, slots, replace=False))
        flat = np.concatenate(flat_parts)
        rng.shuffle(flat)
    record_idx = np.searchsorted(np.cumsum(lengths), flat, side="right")
    previous = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    time_idx = flat - previous[record_idx]
    return record_idx.astype(np.int32), time_idx.astype(np.int32), y[flat].astype(np.float32)


def materialize(records: list[Record], scaled: list[np.ndarray], record_idx: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
    windows = np.zeros((len(record_idx), 16, len(FEATURES) * 2), dtype=np.float32)
    for i, (r_idx, t) in enumerate(zip(record_idx.tolist(), time_idx.tolist())):
        start = max(0, t - 15)
        history = scaled[r_idx][start:t + 1]
        windows[i, 16 - len(history):] = history
    return windows


def summarize_windows(windows: np.ndarray) -> np.ndarray:
    """Causal summary features for compact classical model comparisons."""
    n = len(FEATURES)
    values, observed = windows[:, :, :n], windows[:, :, n:]
    mask = observed > .5
    count = np.maximum(mask.sum(axis=1), 1)
    mean = (values * mask).sum(axis=1) / count
    centered = np.where(mask, values - mean[:, None, :], 0)
    spread = np.sqrt((centered * centered).sum(axis=1) / count)
    last_index = np.maximum(mask.shape[1] - 1 - np.argmax(mask[:, ::-1, :], axis=1), 0)
    first_index = np.argmax(mask, axis=1)
    batch = np.arange(len(windows))[:, None]
    columns = np.arange(n)[None, :]
    last = values[batch, last_index, columns]
    first = values[batch, first_index, columns]
    coverage = mask.mean(axis=1)
    return np.concatenate((last, mean, spread, last - first, coverage), axis=1).astype(np.float32)


def grouped_metrics(y: np.ndarray, probability: np.ndarray, groups: np.ndarray | None = None, seed: int = SEED) -> dict[str, Any]:
    pred = probability >= .5
    result: dict[str, float | int | list[list[int]]] = {
        "windows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix_at_0_5": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if groups is None or len(np.unique(groups)) < 20:
        return result
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    grouped_indices = {group: np.flatnonzero(groups == group) for group in unique}
    auc_values, ap_values = [], []
    for _ in range(100):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        sampled_indices = np.concatenate([grouped_indices[group] for group in sampled_groups])
        sampled_y, sampled_probability = y[sampled_indices], probability[sampled_indices]
        if sampled_y.min() == sampled_y.max():
            continue
        auc_values.append(roc_auc_score(sampled_y, sampled_probability))
        ap_values.append(average_precision_score(sampled_y, sampled_probability))
    if auc_values:
        result["patient_bootstrap_95pct_roc_auc"] = [float(np.percentile(auc_values, 2.5)), float(np.percentile(auc_values, 97.5))]
        result["patient_bootstrap_95pct_average_precision"] = [float(np.percentile(ap_values, 2.5)), float(np.percentile(ap_values, 97.5))]
        result["bootstrap_patients"] = int(len(unique))
    return result


def metrics(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, groups: np.ndarray | None = None) -> dict[str, Any]:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            batch = torch.from_numpy(x[start:start + 1024]).to(device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    probability = np.concatenate(scores)
    return grouped_metrics(y, probability, groups)


def logistic_metrics(model: Any, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> dict[str, Any]:
    probability = model.predict_proba(x)[:, 1]
    return grouped_metrics(y, probability, groups)


def train(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data_root = args.data.resolve()
    set_a = records_in(data_root / "training_setA", "development")
    set_b = records_in(data_root / "training_setB", "external_test")
    if len(set_a) < 100 or len(set_b) < 100:
        raise SystemExit("Expected both public training_setA and training_setB folders. Run scripts/download_physionet_sepsis.py first.")
    order = rng.permutation(len(set_a))
    validation_size = max(1, int(len(set_a) * .2))
    validation_ids = set(order[:validation_size].tolist())
    val_records = [r for i, r in enumerate(set_a) if i in validation_ids]
    train_records = [r for i, r in enumerate(set_a) if i not in validation_ids]
    medians, means, stds = fit_scaling(train_records)
    train_x = [build_features(r, medians, means, stds) for r in train_records]
    val_x = [build_features(r, medians, means, stds) for r in val_records]
    test_x = [build_features(r, medians, means, stds) for r in set_b]

    train_i, train_t, train_y = choose_coordinates(train_records, args.train_windows, rng, balanced=True)
    val_i, val_t, val_y = choose_coordinates(val_records, args.eval_windows, rng)
    test_i, test_t, test_y = choose_coordinates(set_b, args.eval_windows, rng)
    x_train = materialize(train_records, train_x, train_i, train_t)
    x_val = materialize(val_records, val_x, val_i, val_t)
    x_test = materialize(set_b, test_x, test_i, test_t)

    # Two classical comparators use the same causal window and patient split.
    summary_train, summary_val, summary_test = map(summarize_windows, (x_train, x_val, x_test))
    logistic = LogisticRegression(C=.1, class_weight="balanced", max_iter=160, solver="lbfgs", random_state=SEED)
    logistic.fit(summary_train, train_y)
    val_logistic_metrics = logistic_metrics(logistic, summary_val, val_y, val_i)
    external_logistic_metrics = logistic_metrics(logistic, summary_test, test_y, test_i)
    boosted = HistGradientBoostingClassifier(max_iter=75, learning_rate=.08, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=1.0, random_state=SEED)
    boosted.fit(summary_train, train_y)
    val_boosted_metrics = logistic_metrics(boosted, summary_val, val_y, val_i)
    external_boosted_metrics = logistic_metrics(boosted, summary_test, test_y, test_i)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CausalSepsisNet().to(device)
    positive = max(1.0, float(np.sum(train_y)))
    negative = max(1.0, float(len(train_y) - positive))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(4.0, np.sqrt(negative / positive))], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_y))
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    best_ap = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = metrics(model, x_val, val_y, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_average_precision": float(val_metrics["average_precision"])})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} val_auc={val_metrics['roc_auc']:.4f} val_ap={val_metrics['average_precision']:.4f}", flush=True)
        if float(val_metrics["average_precision"]) > best_ap:
            best_ap = float(val_metrics["average_precision"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch >= 3 and history[-1]["validation_average_precision"] < max(h["validation_average_precision"] for h in history[:-1]) - .015:
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint was produced.")
    model.load_state_dict(best_state)
    val_metrics = metrics(model, x_val, val_y, device, val_i)
    external_metrics = metrics(model, x_test, test_y, device, test_i)

    model_dir = ROOT / "models"
    asset_dir = ROOT / "assets"
    model_dir.mkdir(exist_ok=True)
    asset_dir.mkdir(exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "mean": means.tolist(),
        "std": stds.tolist(),
        "median": medians.tolist(),
        "feature_names": list(FEATURE_KEYS),
        "window_hours": 16,
        "label_name": "PhysioNet Challenge 2019 SepsisLabel (six-hour shifted benchmark label)",
        "model_version": "plumoring-sepsis-multisignal-0.2",
    }
    joblib.dump(logistic, model_dir / "sepsis_logistic.joblib", compress=3)
    joblib.dump(boosted, model_dir / "sepsis_histgb.joblib", compress=3)
    validation_comparison = {
        "temporal_conv_bigru_attention": val_metrics,
        "logistic_regression_summary": val_logistic_metrics,
        "hist_gradient_boosting_summary": val_boosted_metrics,
    }
    selected_model = max(validation_comparison, key=lambda name: float(validation_comparison[name]["average_precision"]))
    selected_labels = {
        "temporal_conv_bigru_attention": "Temporal Conv + BiGRU + attention",
        "logistic_regression_summary": "Logistic regression baseline",
        "hist_gradient_boosting_summary": "Histogram gradient-boosting baseline",
    }
    checkpoint["selected_model"] = selected_model
    torch.save(checkpoint, model_dir / "sepsis_risk.pt")
    report = {
        "model_version": checkpoint["model_version"],
        "task": "experimental early sepsis risk research on hospital-recorded vital signs",
        "dataset": "PhysioNet Computing in Cardiology Challenge 2019",
        "data_source": "set A for patient-disjoint development/validation; set B for held-out external evaluation",
        "input_signals": list(FEATURE_KEYS),
        "input_feature_count": len(FEATURES),
        "window_hours": 16,
        "train_patients": len(train_records),
        "validation_patients": len(val_records),
        "external_test_patients": len(set_b),
        "sampled_train_patients": int(len(np.unique(train_i))),
        "sampled_validation_patients": int(len(np.unique(val_i))),
        "sampled_external_test_patients": int(len(np.unique(test_i))),
        "train_sampled_windows": int(len(train_y)),
        "validation_sampled_windows": int(len(val_y)),
        "external_test_sampled_windows": int(len(test_y)),
        "validation": val_metrics,
        "external_test_setB": external_metrics,
        "model_comparison": {
            "temporal_conv_bigru_attention": {"validation": val_metrics, "external_test_setB": external_metrics},
            "logistic_regression_16h_window": {"validation": val_logistic_metrics, "external_test_setB": external_logistic_metrics},
            "hist_gradient_boosting_16h_summary": {"validation": val_boosted_metrics, "external_test_setB": external_boosted_metrics},
        },
        "selected_model_by_setA_validation_average_precision": {"key": selected_model, "label": selected_labels[selected_model]},
        "training_history": history,
        "limits": [
            "The label is the PhysioNet benchmark six-hour-shifted sepsis label; it is not a pulmonary diagnosis.",
            "Hospital vital signs are not smart-ring sensor data. This model does not establish ring-to-disease transfer.",
            "Scores are not calibrated or clinically validated and must not guide care.",
            "Evaluation uses a seeded sample of windows; set B is not the official hidden challenge test set.",
        ],
    }
    (model_dir / "sepsis_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    # One de-identified set-B replay timeline. No subject file name or identifiers are exported.
    candidates = [(i, r) for i, r in enumerate(set_b) if len(r.labels) >= 24 and r.labels.sum() > 0]
    if not candidates:
        candidates = [(i, r) for i, r in enumerate(set_b) if len(r.labels) >= 24]
    replay_idx, replay_record = min(candidates, key=lambda pair: (abs(len(pair[1].labels) - 48), -int(pair[1].labels.sum())))
    replay_features = test_x[replay_idx]
    stop = len(replay_record.labels)
    start = max(0, stop - 72)
    replay_windows = np.zeros((stop - start, 16, len(FEATURES) * 2), np.float32)
    for j, t in enumerate(range(start, stop)):
        first = max(0, t - 15)
        hist = replay_features[first:t + 1]
        replay_windows[j, 16 - len(hist):] = hist
    model.eval()
    with torch.inference_mode():
        replay_scores = torch.sigmoid(model(torch.from_numpy(replay_windows).to(device))).cpu().numpy()
    replay_summaries = summarize_windows(replay_windows)
    replay_baseline_scores = logistic.predict_proba(replay_summaries)[:, 1]
    replay_boosted_scores = boosted.predict_proba(replay_summaries)[:, 1]
    selected_scores = {"temporal_conv_bigru_attention": replay_scores, "logistic_regression_summary": replay_baseline_scores, "hist_gradient_boosting_summary": replay_boosted_scores}[selected_model]
    replay_values = replay_record.values[start:stop]
    replay_observed = replay_record.observed[start:stop]
    replay = {
        "title": "De-identified held-out hospital replay",
        "dataset": "PhysioNet Challenge 2019 · training_setB external holdout",
        "time_step": "one recorded ICU hour",
        "model": checkpoint["model_version"],
        "selected_model": selected_labels[selected_model],
        "input_features": list(FEATURES),
        "rows": [
            {
                "hour": i + 1,
                "hr": None if not replay_observed[i, 0] else round(float(replay_values[i, 0]), 1),
                "spo2": None if not replay_observed[i, 1] else round(float(replay_values[i, 1]), 1),
                "resp": None if not replay_observed[i, 2] else round(float(replay_values[i, 2]), 1),
                "risk_score": round(float(replay_scores[i]), 5),
                "baseline_score": round(float(replay_baseline_scores[i]), 5),
                "boosted_score": round(float(replay_boosted_scores[i]), 5),
                "selected_score": round(float(selected_scores[i]), 5),
                "clinical_inputs": {name: None if not replay_observed[i, j] else round(float(replay_values[i, j]), 3) for j, name in enumerate(FEATURES)},
                "benchmark_label": int(replay_record.labels[start + i]),
                "observed": replay_observed[i].astype(bool).tolist(),
            }
            for i in range(stop - start)
        ],
        "notice": "Recorded hospital data replay, not live ring data. Risk score is experimental and not a diagnosis or clinical alert.",
    }
    (asset_dir / "research_replay.json").write_text(json.dumps(replay, separators=(",", ":")) + "\n")
    print(json.dumps({"validation": validation_comparison, "external_test_setB": {"temporal": external_metrics, "logistic": external_logistic_metrics, "boosted": external_boosted_metrics}, "selected_model": selected_labels[selected_model], "weights": str(model_dir / "sepsis_risk.pt")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/physionet_sepsis_2019"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--train-windows", type=int, default=120_000)
    parser.add_argument("--eval-windows", type=int, default=50_000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
