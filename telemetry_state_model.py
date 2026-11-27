"""Safe, explicitly research-only adapter for the supplied five-state Keras model."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "temporal_telemetry_state_classifier.keras"
MODEL_LABELS = (
    "NORMAL RESTING BASELINE",
    "DAILY ACTIVITY / EXERCISE IMPACT",
    "SUDDEN BREATHING TROUBLE / STRESS",
    "FEVER / LUNG INFECTION ALERT",
    "LOOSE RING / BAD READING (IGNORED)",
)
INPUT_NAMES = (
    "heart_rate_bpm",
    "spo2_percent",
    "respiration_rate_bpm",
    "acoustic_wheeze_index",
    "skin_temperature_delta_c",
    "imu_acceleration_g",
    "eda_index",
    "pm25_ug_m3",
)
INPUT_RANGES = (
    (40.0, 190.0),
    (70.0, 100.0),
    (8.0, 40.0),
    (0.0, 10.0),
    (0.0, 2.5),
    (-3.0, 3.0),
    (0.0, 1.0),
    (0.0, 250.0),
)

_model: Any | None = None
_model_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    """Raised when the optional model runtime or saved model is unavailable."""


def runtime_status() -> dict[str, Any]:
    tensorflow_available = importlib.util.find_spec("tensorflow") is not None
    return {
        "model_file_present": MODEL_PATH.is_file(),
        "tensorflow_available": tensorflow_available,
        "ready_for_research_preview": MODEL_PATH.is_file() and tensorflow_available,
        "input_shape": [16, 8],
        "output_classes": list(MODEL_LABELS),
        "training_dataset_in_bundle": False,
        "validation_report_in_bundle": False,
        "diagnostic_model": False,
    }


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.is_file():
            raise ModelUnavailable("The supplied model file is not installed in this project.")
        if importlib.util.find_spec("tensorflow") is None:
            raise ModelUnavailable("TensorFlow is not installed in this Python environment.")
        try:
            import keras

            loaded = keras.saving.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=True,
            )
        except Exception as exc:
            raise ModelUnavailable(f"The saved model could not be loaded: {exc}") from exc
        input_shape = tuple(loaded.input_shape[1:])
        output_shape = tuple(loaded.output_shape[1:])
        if input_shape != (16, 8) or output_shape != (5,):
            raise ModelUnavailable(
                f"Unexpected saved model shapes: input={input_shape}, output={output_shape}."
            )
        _model = loaded
    return _model


def _validated_inputs(payload: dict[str, Any]) -> tuple[np.ndarray, str, np.ndarray]:
    if "sequence" in payload:
        try:
            values = np.asarray(payload["sequence"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Sequence must contain numeric sensor readings.") from exc
        if values.shape != (16, 8):
            raise ValueError("Sequence must contain exactly 16 rows of 8 sensor readings.")
        mode = "16 observed steps supplied by caller"
    else:
        try:
            snapshot = np.asarray(payload.get("sensors"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Sensors must contain eight numeric readings.") from exc
        if snapshot.shape != (8,):
            raise ValueError("Sensors must contain exactly eight readings in the documented order.")
        values = np.repeat(snapshot[np.newaxis, :], 16, axis=0)
        mode = "single snapshot repeated across 16 steps, matching the supplied demo adapter"

    if not np.isfinite(values).all():
        raise ValueError("Sensor readings must all be finite numbers.")
    for index, (minimum, maximum) in enumerate(INPUT_RANGES):
        column = values[:, index]
        if np.any(column < minimum) or np.any(column > maximum):
            raise ValueError(
                f"{INPUT_NAMES[index]} is outside the supplied model interface range "
                f"({minimum:g} to {maximum:g})."
            )

    # Match the normalization in the supplied app.py exactly.
    normalized = np.empty_like(values, dtype=np.float32)
    normalized[:, 0] = values[:, 0] / 200.0
    normalized[:, 1] = values[:, 1] / 100.0
    normalized[:, 2] = values[:, 2] / 40.0
    normalized[:, 3] = values[:, 3] / 10.0
    normalized[:, 4] = values[:, 4] / 2.5
    normalized[:, 5] = (values[:, 5] + 3.0) / 6.0
    normalized[:, 6] = values[:, 6] / 1.0
    normalized[:, 7] = values[:, 7] / 250.0
    return normalized[np.newaxis, ...], mode, values


def _rule_hypothesis(values: np.ndarray) -> dict[str, Any]:
    """Map readings to an explicitly unvalidated, non-probabilistic demo pattern."""
    hr, spo2, rr, wheeze, temp, imu, eda, pm25 = (float(value) for value in values)
    observations = []
    if spo2 < 95:
        observations.append(f"SpO₂ input {spo2:.1f}%")
    if hr > 100:
        observations.append(f"heart-rate input {hr:.0f} bpm")
    if rr > 22:
        observations.append(f"breathing-rate input {rr:.0f}/min")
    if wheeze >= 3:
        observations.append(f"companion wheeze index {wheeze:.1f}/10")
    if temp >= 1:
        observations.append(f"skin-temperature delta {temp:+.2f} °C")
    if abs(imu) >= 1:
        observations.append(f"motion input {imu:.2f} g")
    if eda >= 0.7:
        observations.append(f"EDA input {eda:.2f}")
    if pm25 >= 50:
        observations.append(f"context PM2.5 input {pm25:.0f} µg/m³")

    label = "No supplied rule pattern triggered"
    detail = "This does not rule out a pulmonary condition."

    # These hand-written branches are separate from Model 1's learned state
    # classes. They are demonstrations only, not validated disease criteria.
    if abs(imu) > 2.2:
        label = "High-motion snapshot · pulmonary pattern withheld"
        detail = (
            "This single reading has very high motion. The experimental rules withhold a named "
            "pulmonary pattern because movement can dominate a spot snapshot; this is not a "
            "validated signal-quality assessment."
        )
    elif temp >= 1.2 and (wheeze >= 3.5 or rr >= 22 or spo2 < 93):
        label = "Readings may resemble a respiratory-infection pattern"
        detail = (
            "This experimental screen cannot confirm pneumonia, bronchitis, or another infection. "
            "If breathing symptoms persist or concern you, consider a clinician check-up."
        )
    elif wheeze >= 4.0 and spo2 < 94:
        if pm25 >= 80:
            label = "Airway-symptom pattern · high PM2.5 context"
            detail = "The readings overlap with an airway-irritation pattern; PM2.5 is context, not a ring measurement. Consider a check-up if symptoms persist."
        elif spo2 < 91 and rr >= 22 and wheeze < 5.0:
            label = "Readings may resemble a COPD-like pattern"
            detail = "This screen cannot distinguish COPD from asthma or other causes. If breathing symptoms persist or concern you, consider a clinician check-up."
        else:
            label = "Readings may resemble an asthma-like wheeze pattern"
            detail = "This is not an asthma diagnosis; COPD and other causes can overlap. If breathing symptoms persist or concern you, consider a clinician check-up."
    elif spo2 < 92 and rr >= 22:
        label = "Low-oxygen + rapid-breathing pattern · cause unclear"
        detail = "This flags a reading combination only; the readings do not identify a pulmonary disease."
    elif wheeze >= 3.5:
        label = "Wheeze pattern · cause unclear"
        detail = "Wheeze is a companion input, not a ring sensor, and does not identify its cause."
    elif temp >= 1.2:
        label = "Elevated skin-temperature pattern · cause unclear"
        detail = "A skin-temperature delta is not a core-temperature measurement and cannot establish fever or infection."
    elif rr > 24:
        label = "Rapid-breathing pattern · cause unclear"
        detail = "This reading alone does not identify a pulmonary disease."

    return {
        "label": label,
        "detail": detail,
        "observations": observations or ["No supplied-rule threshold crossed"],
        "source": "Unvalidated hand-written demo thresholds, independent of Model 1",
        "probability": None,
        "is_diagnosis": False,
    }


def predict_state(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    model_input, input_mode, raw_values = _validated_inputs(payload)
    model = _load_model()
    try:
        raw_scores = model(model_input, training=False)
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ModelUnavailable(f"The model could not process this input: {exc}") from exc
    if scores.shape != (5,) or not np.isfinite(scores).all():
        raise ModelUnavailable("The model returned an invalid output vector.")

    top_index = int(np.argmax(scores))
    return {
        "model_id": "plumoring-temporal-se-resnet-bigru-attention",
        "model_output_type": "five-class telemetry-state pattern",
        "input_mode": input_mode,
        "top_class_index": top_index,
        "top_class_label": MODEL_LABELS[top_index],
        "top_class_raw_score": round(float(scores[top_index]), 6),
        "scores_are_calibrated_probabilities": False,
        "rule_hypothesis": _rule_hypothesis(raw_values[-1]),
        "class_scores": [
            {"index": index, "label": label, "raw_score": round(float(scores[index]), 6)}
            for index, label in enumerate(MODEL_LABELS)
        ],
        "diagnosis": None,
        "notice": (
            "Experimental output from the supplied model. Its training dataset and "
            "held-out validation report were not included. The class labels are not a "
            "pulmonary diagnosis, and the raw scores are not calibrated probabilities."
        ),
    }
