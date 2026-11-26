"""Local, research-only PlumoRing AI web app and inference API."""

from __future__ import annotations

import importlib.util
import math
import threading
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sepsis_risk.pt"
BASELINE_PATH = ROOT / "models" / "sepsis_logistic.joblib"
BOOSTED_PATH = ROOT / "models" / "sepsis_histgb.joblib"
METRICS_PATH = ROOT / "models" / "sepsis_metrics.json"
REPLAY_PATH = ROOT / "assets" / "research_replay.json"
WEB_PAGES = {"favicon.svg", "responsive.css", "pages.css", "privacy.html", "settings.html", "404.html"}
WINDOW_HOURS = 16

app = Flask(__name__, static_folder=None)
_model: Any | None = None
_baseline: Any | None = None
_boosted: Any | None = None
_checkpoint: dict[str, Any] | None = None
_model_lock = threading.Lock()


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(ROOT / "assets", filename)


@app.get("/models/<path:filename>")
def model_artifacts(filename: str):
    if filename != "sepsis_metrics.json":
        abort(404)
    return send_from_directory(ROOT / "models", filename)


@app.get("/<path:filename>")
def pages(filename: str):
    if filename not in WEB_PAGES:
        abort(404)
    return send_from_directory(ROOT, filename)


def health_payload() -> dict[str, Any]:
    runtime_present = importlib.util.find_spec("torch") is not None
    sklearn_present = importlib.util.find_spec("sklearn") is not None
    try:
        from scripts.train_sepsis_risk import FEATURES
        feature_names = list(FEATURES)
    except ImportError:
        feature_names = []
    report = {}
    if METRICS_PATH.is_file():
        try:
            import json
            report = json.loads(METRICS_PATH.read_text())
        except (OSError, ValueError):
            report = {}
    return {
        "ready": MODEL_PATH.is_file() and BASELINE_PATH.is_file() and BOOSTED_PATH.is_file() and runtime_present and sklearn_present,
        "model_file_present": MODEL_PATH.is_file(),
        "baseline_file_present": BASELINE_PATH.is_file(),
        "boosted_file_present": BOOSTED_PATH.is_file(),
        "runtime_present": runtime_present,
        "comparison_runtime_present": sklearn_present,
        "replay_present": REPLAY_PATH.is_file(),
        "model_version": report.get("model_version", "plumoring-sepsis-multisignal-0.2"),
        "task": "experimental early sepsis risk research on hospital vital signs",
        "input_signals": feature_names,
        "window_hours": WINDOW_HOURS,
        "notices": [
            "Hospital-recorded vitals are not ring data.",
            "Experimental score, not calibrated or clinically validated.",
            "No diagnosis or care recommendation is generated.",
        ],
    }


@app.get("/api/health")
def health():
    return jsonify(health_payload())


@app.get("/api/replay")
def replay():
    if not REPLAY_PATH.is_file():
        return jsonify({"error": "Recorded replay is not ready. Download the public dataset and run the training workflow."}), 503
    return send_from_directory(ROOT / "assets", REPLAY_PATH.name, max_age=3600)


def get_model():
    global _model, _checkpoint
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.is_file():
                    raise RuntimeError("The research model is not trained. See README for the public-data workflow.")
                try:
                    import torch
                    from scripts.train_sepsis_risk import CausalSepsisNet
                except ImportError as exc:
                    raise RuntimeError("PyTorch is unavailable. Install requirements-model.txt.") from exc
                loaded = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
                model = CausalSepsisNet()
                model.load_state_dict(loaded["state_dict"])
                model.eval()
                _checkpoint = loaded
                _model = model
    return _model, _checkpoint


def get_baselines():
    global _baseline, _boosted
    if _baseline is None or _boosted is None:
        if not BASELINE_PATH.is_file() or not BOOSTED_PATH.is_file():
            raise RuntimeError("The comparison models are not trained yet.")
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("scikit-learn is unavailable. Install requirements-model.txt.") from exc
        _baseline = _baseline or joblib.load(BASELINE_PATH)
        _boosted = _boosted or joblib.load(BOOSTED_PATH)
    return _baseline, _boosted


def parse_sequence(raw: Any):
    import numpy as np
    from scripts.train_sepsis_risk import BOUNDS, FEATURES as MODEL_FEATURES

    if not isinstance(raw, list) or len(raw) != WINDOW_HOURS:
        raise ValueError("Send exactly 16 ordered hourly rows matching the hospital model's feature schema.")
    values = np.full((WINDOW_HOURS, len(MODEL_FEATURES)), np.nan, dtype=np.float32)
    aliases = {
        "HR": ("hr", "heart_rate"), "O2Sat": ("spo2", "SpO2", "oxygen_saturation"),
        "Resp": ("resp", "respiration_rate"), "Temp": ("temperature",),
        "SBP": ("sbp",), "MAP": ("map",), "DBP": ("dbp",),
        "Age": ("age",), "ICULOS": ("iculos",),
    }
    for row_i, row in enumerate(raw):
        if isinstance(row, dict):
            row = [next((row[name] for name in (key, *aliases.get(key, ())) if name in row), None) for key in MODEL_FEATURES]
        if not isinstance(row, (list, tuple)) or len(row) != len(MODEL_FEATURES):
            raise ValueError(f"Each row must contain {len(MODEL_FEATURES)} values in the published feature order.")
        for column, (value, key) in enumerate(zip(row, MODEL_FEATURES)):
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Readings must be numbers or null when missing.") from exc
            minimum, maximum = BOUNDS[key]
            if not math.isfinite(number) or not minimum <= number <= maximum:
                raise ValueError(f"{key} is outside its allowed data-validation range ({minimum}–{maximum}).")
            values[row_i, column] = number
    # Ring-only HR/SpO₂/respiration values cannot be fed to this hospital model.
    # Require a meaningful clinical schema, while allowing normal lab missingness.
    ring_features = {"HR", "O2Sat", "Resp"}
    context_features = [name for name in MODEL_FEATURES if name not in ring_features and name not in {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"}]
    context_indices = [MODEL_FEATURES.index(name) for name in context_features]
    visible_context = np.isfinite(values[:, context_indices]).any(axis=0).sum()
    core_indices = [MODEL_FEATURES.index(name) for name in ("HR", "O2Sat", "Resp")]
    if np.isfinite(values[:, core_indices]).sum() < 12 or visible_context < 4:
        raise ValueError("This is a hospital-data model: provide a 16-hour clinical sequence with multiple vital and context channels. Ring-only readings are not supported.")
    return values


@app.post("/api/score-window")
def score_window():
    try:
        raw = request.get_json(silent=True) or {}
        values = parse_sequence(raw.get("sequence"))
        model, checkpoint = get_model()
        import numpy as np
        import torch

        medians = np.asarray(checkpoint["median"], dtype=np.float32)
        means = np.asarray(checkpoint["mean"], dtype=np.float32)
        stds = np.asarray(checkpoint["std"], dtype=np.float32)
        observed = np.isfinite(values)
        filled = values.copy()
        last = medians.copy()
        for t in range(len(filled)):
            present = observed[t]
            last[present] = filled[t, present]
            filled[t, ~present] = last[~present]
        standardized = np.clip((filled - means) / stds, -5, 5)
        model_input = np.concatenate((standardized, observed.astype(np.float32)), axis=1)[None, ...]
        with torch.inference_mode():
            score = float(torch.sigmoid(model(torch.from_numpy(model_input))).item())
        from scripts.train_sepsis_risk import summarize_windows
        baseline, boosted = get_baselines()
        summary = summarize_windows(model_input)
        baseline_score = float(baseline.predict_proba(summary)[0, 1])
        boosted_score = float(boosted.predict_proba(summary)[0, 1])
        model_keys = {
            "temporal_conv_bigru_attention": score,
            "logistic_regression_summary": baseline_score,
            "hist_gradient_boosting_summary": boosted_score,
        }
        selected_key = checkpoint.get("selected_model", "temporal_conv_bigru_attention")
        return jsonify({
            "model_version": checkpoint.get("model_version", "plumoring-sepsis-multisignal-0.2"),
            "temporal_research_score": round(score, 5),
            "logistic_baseline_score": round(baseline_score, 5),
            "boosted_baseline_score": round(boosted_score, 5),
            "selected_model": selected_key,
            "selected_research_score": round(float(model_keys.get(selected_key, score)), 5),
            "window_hours": WINDOW_HOURS,
            "score_kind": "relative benchmark score; not a calibrated probability",
            "input_signals": list(checkpoint.get("feature_names", ())),
            "disclaimer": "Hospital-data research model. It is not ring-validated, not a diagnosis, and not for clinical decisions.",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
