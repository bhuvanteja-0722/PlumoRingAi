"""Local PlumoRing AI research interface and pulmonary-evidence API."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, abort, jsonify, request, send_from_directory

from telemetry_state_model import ModelUnavailable, predict_state, runtime_status


ROOT = Path(__file__).resolve().parent


def _load_local_env() -> None:
    """Load simple KEY=value entries from an ignored local .env file."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].strip()
        if "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        name, value = name.strip(), value.strip()
        if name and name.replace("_", "").isalnum() and name not in os.environ:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[name] = value


_load_local_env()
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
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024

ASSISTANT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6.2-astra")
CHANNEL_RANGES = {
    "hr": (40, 190), "spo2": (70, 100), "rr": (8, 40), "wheeze": (0, 10),
    "temp": (0, 2.5), "imu": (-3, 3), "eda": (0, 1), "pm25": (0, 250),
}


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


def _validated_readings(value: Any, *, limit: int = 12) -> list[dict[str, float]]:
    if not isinstance(value, list) or not value or len(value) > limit:
        raise ValueError(f"Provide between 1 and {limit} saved readings.")
    cleaned: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each saved reading must be an object.")
        values = item.get("readings", item.get("sensors"))
        if isinstance(values, list) and len(values) == len(CHANNEL_RANGES):
            values = dict(zip(CHANNEL_RANGES, values))
        if not isinstance(values, dict) or set(values) != set(CHANNEL_RANGES):
            raise ValueError("Each saved reading must contain the eight supported channels.")
        row: dict[str, float] = {}
        for channel, (minimum, maximum) in CHANNEL_RANGES.items():
            try:
                number = float(values[channel])
            except (TypeError, ValueError):
                raise ValueError(f"The {channel} reading must be numeric.") from None
            if not math.isfinite(number) or number < minimum or number > maximum:
                raise ValueError(f"The {channel} reading is outside the supported range.")
            row[channel] = number
        cleaned.append(row)
    return cleaned


def _openai_api_key() -> str | None:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if not value or value.lower().startswith(("replace-with-", "your-api-key")):
        return None
    return value


def _openai_text(instructions: str, input_value: Any, *, max_output_tokens: int = 360) -> str:
    api_key = _openai_api_key()
    if not api_key:
        raise RuntimeError("AI features are not configured. Add OPENAI_API_KEY to the local .env file and restart the app.")
    body = json.dumps({
        "model": ASSISTANT_MODEL,
        "instructions": instructions,
        "input": input_value,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }).encode("utf-8")
    req = Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=35) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError("The OpenAI API key was not accepted. Check the key and its project access.") from None
        if exc.code == 429:
            raise RuntimeError("The OpenAI API rate limit or account quota was reached. Check the API project billing and limits.") from None
        raise RuntimeError(f"The OpenAI API could not complete this request (HTTP {exc.code}).") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Could not reach the OpenAI API. Check the internet connection and try again.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("The OpenAI API returned an unreadable response.") from None
    output: list[str] = []
    for item in result.get("output", []) if isinstance(result, dict) else []:
        if item.get("type") == "message":
            output.extend(part.get("text", "") for part in item.get("content", []) if part.get("type") == "output_text")
    answer = "\n".join(part for part in output if part).strip()
    if not answer:
        raise RuntimeError("The OpenAI API did not return a text answer. Please try again.")
    return answer[:5000]


@app.get("/api/assistant/status")
def assistant_status():
    return jsonify({"provider": "OpenAI API", "configured": _openai_api_key() is not None, "model": ASSISTANT_MODEL})


@app.post("/api/assistant/insights")
def assistant_insights():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        readings = _validated_readings(payload.get("readings"))
        context = json.dumps(readings, separators=(",", ":"))
        answer = _openai_text(
            "You are an educational assistant inside a pulmonary research prototype. Summarize only observable changes in the supplied numeric readings. Do not diagnose, name a likely disease, give probabilities, classify severity, or recommend treatment or medication. These are manually entered demo snapshots, not validated ring data, and the model has no pulmonary disease labels. Give a brief neutral trend summary and general low-risk lung-health habits; say that persistent or concerning symptoms merit a clinician check-up. If the data are too sparse, say so. Do not infer age, sex, height, weight, or identity.",
            "Summarize these recent readings as a short, plain-language research preview.\n" + context,
            max_output_tokens=300,
        )
        return jsonify({"answer": answer, "model": ASSISTANT_MODEL})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        code = 503 if _openai_api_key() is None else 502
        return jsonify({"error": str(exc)}), code


@app.post("/api/assistant/chat")
def assistant_chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 12:
        return jsonify({"error": "Send between 1 and 12 recent chat messages."}), 400
    include_history = bool(payload.get("include_readings"))
    message_limit = 4000 if include_history else 1200
    cleaned_messages: list[dict[str, str]] = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            return jsonify({"error": "Chat messages must have a user or assistant role."}), 400
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > message_limit:
            return jsonify({"error": f"Each chat message must contain 1 to {message_limit:,} characters."}), 400
        total_chars += len(content)
        cleaned_messages.append({"role": message["role"], "content": content.strip()})
    if total_chars > 6000:
        return jsonify({"error": "The recent chat is too long. Clear the chat or shorten the messages."}), 400
    readings_context = ""
    if include_history:
        try:
            readings = _validated_readings(payload.get("readings"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        readings_context = "\nThe user explicitly included recent demo readings. They are not diagnostic or validated ring measurements. Discuss only their numeric trend, and do not infer a disease: " + json.dumps(readings, separators=(",", ":"))
        for message in reversed(cleaned_messages):
            if message["role"] == "user":
                message["content"] += readings_context
                break
    try:
        answer = _openai_text(
            "You are PlumoRing AI's educational pulmonary-research chatbot. Answer questions about the project, respiratory health concepts, and general wellness in concise, accessible language. Never diagnose, tell someone they may have a specific disease based on these inputs, estimate disease probability, provide treatment/medication instructions, or say the prototype detected a condition. Model 1 returns uncalibrated telemetry-state classes and is not a pulmonary disease model. The app's demo rule labels are unvalidated and cannot establish disease. If asked about personal symptoms, explain this limitation and suggest speaking with a qualified clinician for persistent or concerning symptoms. For severe breathing difficulty, blue/grey lips, fainting, or other emergency symptoms, advise contacting local emergency services now. Do not infer age, sex, height, weight, or identity. No medical tools are available.",
            cleaned_messages,
            max_output_tokens=450,
        )
        return jsonify({"answer": answer, "model": ASSISTANT_MODEL})
    except RuntimeError as exc:
        code = 503 if _openai_api_key() is None else 502
        return jsonify({"error": str(exc)}), code


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
