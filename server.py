"""Local PlumoRing AI research interface and pulmonary-evidence API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, send_from_directory


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "pulmonary_feasibility_metrics.json"
COPD_PPG_METRICS_PATH = ROOT / "models" / "copd_ppg_screening_metrics.json"
AAMOS_METRICS_PATH = ROOT / "models" / "aamos_asthma_worsening_metrics.json"
OSASUD_METRICS_PATH = ROOT / "models" / "osasud_ppg_event_metrics.json"
TOLIFE_TREND_METRICS_PATH = ROOT / "models" / "tolife_personalized_trend_metrics.json"
MODEL_REPORTS = {
    METRICS_PATH.name: METRICS_PATH,
    COPD_PPG_METRICS_PATH.name: COPD_PPG_METRICS_PATH,
    AAMOS_METRICS_PATH.name: AAMOS_METRICS_PATH,
    OSASUD_METRICS_PATH.name: OSASUD_METRICS_PATH,
    TOLIFE_TREND_METRICS_PATH.name: TOLIFE_TREND_METRICS_PATH,
}
WEB_PAGES = {"favicon.svg", "responsive.css", "pages.css", "privacy.html", "settings.html", "404.html"}
ASSET_ALLOWLIST = {"anatomical-heart.glb", "visible-human-lungs.glb"}

app = Flask(__name__, static_folder=None)


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/assets/<path:filename>")
def assets(filename: str):
    if filename not in ASSET_ALLOWLIST:
        abort(404)
    return send_from_directory(ROOT / "assets", filename)


@app.get("/models/<path:filename>")
def model_artifacts(filename: str):
    report_path = MODEL_REPORTS.get(filename)
    if report_path is None:
        abort(404)
    if not report_path.is_file():
        return jsonify({"error": "Pulmonary feasibility evaluation has not been run yet."}), 503
    return send_from_directory(report_path.parent, report_path.name)


@app.get("/<path:filename>")
def pages(filename: str):
    if filename not in WEB_PAGES:
        abort(404)
    return send_from_directory(ROOT, filename)


def _metrics() -> dict[str, Any]:
    if not METRICS_PATH.is_file():
        return {}
    try:
        value = json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


@app.get("/api/health")
def health():
    report = _metrics()
    return jsonify({
        "ready": False,
        "model_version": report.get("model_version", "pulmonary-evidence-pending"),
        "task": "separate research-only COPD flare, asthma worsening, and overnight apnea/hypopnea event feasibility studies",
        "disease_diagnosis_supported": [],
        "candidate_meets_exploratory_gate": report.get("candidate_meets_exploratory_gate", False),
        "candidate_gate_scope": "TOLIFE COPD exacerbation forecasting only; this is not a diagnosis or an overall project readiness check.",
        "deployment_status": "not_ready",
        "ring_validated": False,
        "continuous_ring_stream": False,
        "notices": [
            "No pulmonary disease diagnosis or patient-facing risk score is available.",
            "The public COPD cohort uses smartwatch, mattress, mobility and spirometer-related sensors, not this proposed ring.",
            "The current grouped evaluation did not beat the no-skill reference; do not use for medical decisions.",
            "The separate PPG experiment has six independent people and is not a validated COPD detector.",
            "The AAMOS-00 asthma result is retrospective weekly classification and fails the held-out sensitivity gate.",
            "OSASUD showed an exploratory signal for annotated apnea/hypopnea windows, but model selection is optimistic; it is not a sleep-apnea diagnosis and uses hospital finger PPG, not this ring.",
            "The personalized TOLIFE trend baseline also failed its no-skill comparison on a small eligible subset; no alert is available.",
            "A larger 2026 continuous COPD wearable study has restricted participant data and has not been evaluated by PlumoRing AI.",
        ],
    })


@app.get("/api/pulmonary/metrics")
def pulmonary_metrics():
    report = _metrics()
    if not report:
        return jsonify({"error": "Pulmonary feasibility evaluation has not been run yet."}), 503
    return jsonify(report)


@app.get("/api/pulmonary/copd-ppg-metrics")
def copd_ppg_metrics():
    if not COPD_PPG_METRICS_PATH.is_file():
        return jsonify({"error": "The COPD/PPG feasibility evaluation has not been run yet."}), 503
    try:
        report = json.loads(COPD_PPG_METRICS_PATH.read_text())
    except (OSError, ValueError):
        return jsonify({"error": "The COPD/PPG feasibility report is unavailable."}), 503
    return jsonify(report)


@app.get("/api/pulmonary/aamos-metrics")
def aamos_metrics():
    if not AAMOS_METRICS_PATH.is_file():
        return jsonify({"error": "The AAMOS-00 asthma-worsening evaluation has not been run yet."}), 503
    try:
        report = json.loads(AAMOS_METRICS_PATH.read_text())
    except (OSError, ValueError):
        return jsonify({"error": "The AAMOS-00 aggregate report is unavailable."}), 503
    return jsonify(report)


@app.get("/api/pulmonary/osasud-metrics")
def osasud_metrics():
    if not OSASUD_METRICS_PATH.is_file():
        return jsonify({"error": "The OSASUD PPG event evaluation has not been run yet."}), 503
    try:
        report = json.loads(OSASUD_METRICS_PATH.read_text())
    except (OSError, ValueError):
        return jsonify({"error": "The OSASUD aggregate report is unavailable."}), 503
    return jsonify(report)


@app.get("/api/pulmonary/tolife-trend-metrics")
def tolife_trend_metrics():
    if not TOLIFE_TREND_METRICS_PATH.is_file():
        return jsonify({"error": "The TOLIFE personalized trend evaluation has not been run yet."}), 503
    try:
        report = json.loads(TOLIFE_TREND_METRICS_PATH.read_text())
    except (OSError, ValueError):
        return jsonify({"error": "The TOLIFE personalized trend report is unavailable."}), 503
    return jsonify(report)


@app.get("/api/replay")
def legacy_replay():
    return jsonify({"error": "The retired hospital replay is not part of this pulmonary research workspace."}), 410


@app.post("/api/score-window")
def legacy_score_window():
    return jsonify({"error": "No patient-facing pulmonary inference endpoint is available. The COPD and asthma studies failed their research gates; OSASUD only screened overnight hospital event windows and does not diagnose a disease or validate this ring."}), 410


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
