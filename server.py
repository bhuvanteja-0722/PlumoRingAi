"""Local PlumoRing AI research interface and pulmonary-evidence API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory

from telemetry_state_model import ModelUnavailable, predict_state, runtime_status


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "pulmonary_feasibility_metrics.json"
COPD_PPG_METRICS_PATH = ROOT / "models" / "copd_ppg_screening_metrics.json"
AAMOS_METRICS_PATH = ROOT / "models" / "aamos_asthma_worsening_metrics.json"
OSASUD_METRICS_PATH = ROOT / "models" / "osasud_ppg_event_metrics.json"
TOLIFE_TREND_METRICS_PATH = ROOT / "models" / "tolife_personalized_trend_metrics.json"
HEALTHRING_METRICS_PATH = ROOT / "models" / "healthring_ring_vitals_metrics.json"
BIDMC_RR_METRICS_PATH = ROOT / "models" / "bidmc_ppg_respiratory_rate_metrics.json"
TOLIFE_ASSOCIATION_PATH = ROOT / "models" / "tolife_clinical_associations.json"
MODEL_REPORTS = {
    METRICS_PATH.name: METRICS_PATH,
    COPD_PPG_METRICS_PATH.name: COPD_PPG_METRICS_PATH,
    AAMOS_METRICS_PATH.name: AAMOS_METRICS_PATH,
    OSASUD_METRICS_PATH.name: OSASUD_METRICS_PATH,
    TOLIFE_TREND_METRICS_PATH.name: TOLIFE_TREND_METRICS_PATH,
    HEALTHRING_METRICS_PATH.name: HEALTHRING_METRICS_PATH,
    BIDMC_RR_METRICS_PATH.name: BIDMC_RR_METRICS_PATH,
    TOLIFE_ASSOCIATION_PATH.name: TOLIFE_ASSOCIATION_PATH,
}
WEB_PAGES = {"favicon.svg", "responsive.css", "pages.css", "privacy.html", "settings.html", "404.html"}
ASSET_ALLOWLIST = {"anatomical-heart.glb", "visible-human-lungs.glb"}

app = Flask(__name__, static_folder=None)


@app.after_request
def prevent_stale_html(response):
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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
    model_1_status = runtime_status()
    return jsonify({
        "ready": False,
        "model_version": report.get("model_version", "pulmonary-evidence-pending"),
        "task": "separate research-only pulmonary-event feasibility studies and PPG/ring vital-estimation benchmarks",
        "disease_diagnosis_supported": [],
        "candidate_meets_exploratory_gate": report.get("candidate_meets_exploratory_gate", False),
        "candidate_gate_scope": "TOLIFE COPD exacerbation forecasting only; this is not a diagnosis or an overall project readiness check.",
        "deployment_status": "not_ready",
        "ring_validated": False,
        "continuous_ring_stream": False,
        "model_1_research_preview": model_1_status,
        "notices": [
            "No pulmonary disease diagnosis or patient-facing risk score is available.",
            "HealthRing provides partial-subset ring-vital estimation only; it has no pulmonary disease labels.",
            "The BIDMC respiratory-rate model did not transfer to HealthRing better than a mean baseline.",
            "The public COPD cohort uses smartwatch, mattress, mobility and spirometer-related sensors, not this proposed ring.",
            "The current grouped evaluation did not beat the no-skill reference; do not use for medical decisions.",
            "The separate PPG experiment has six independent people and is not a validated COPD detector.",
            "The AAMOS-00 asthma result is retrospective weekly classification and fails the held-out sensitivity gate.",
            "OSASUD showed an exploratory signal for annotated apnea/hypopnea windows, but model selection is optimistic; it is not a sleep-apnea diagnosis and uses hospital finger PPG, not this ring.",
            "The personalized TOLIFE trend baseline also failed its no-skill comparison on a small eligible subset; no alert is available.",
            "A larger 2026 continuous COPD wearable study has restricted participant data and has not been evaluated by PlumoRing AI.",
        ],
    })


@app.get("/api/models/temporal-telemetry/status")
# Preserve API paths used by pages that were already open before this rename.
@app.get("/api/models/friend-state/status")
def temporal_telemetry_status():
    status = runtime_status()
    status["model_id"] = "plumoring-temporal-se-resnet-bigru-attention"
    status["model_output_type"] = "five-class telemetry-state pattern"
    return jsonify(status)


@app.post("/api/models/temporal-telemetry/predict")
# Preserve API paths used by pages that were already open before this rename.
@app.post("/api/models/friend-state/predict")
def temporal_telemetry_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        return jsonify(predict_state(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ModelUnavailable as exc:
        return jsonify({"error": str(exc), "status": runtime_status()}), 503


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


def _serve_aggregate_report(path: Path, label: str):
    if not path.is_file():
        return jsonify({"error": f"The {label} aggregate report has not been generated yet."}), 503
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return jsonify({"error": f"The {label} aggregate report is unavailable."}), 503
    return jsonify(report)


@app.get("/api/pulmonary/healthring-vitals-metrics")
def healthring_vitals_metrics():
    return _serve_aggregate_report(HEALTHRING_METRICS_PATH, "HealthRing vitals")


@app.get("/api/pulmonary/bidmc-rr-metrics")
def bidmc_rr_metrics():
    return _serve_aggregate_report(BIDMC_RR_METRICS_PATH, "BIDMC respiratory-rate")


@app.get("/api/pulmonary/tolife-associations")
def tolife_associations():
    return _serve_aggregate_report(TOLIFE_ASSOCIATION_PATH, "TOLIFE clinical-association")


@app.get("/api/replay")
def legacy_replay():
    return jsonify({"error": "The retired hospital replay is not part of this pulmonary research workspace."}), 410


@app.post("/api/score-window")
def legacy_score_window():
    return jsonify({"error": "No patient-facing pulmonary inference endpoint is available. The COPD and asthma studies failed their research gates; OSASUD only screened overnight hospital event windows and does not diagnose a disease or validate this ring."}), 410


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
