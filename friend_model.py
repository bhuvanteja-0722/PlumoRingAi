"""Safe, explicitly research-only adapter for the supplied five-state Keras model."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "friend_telemetry_state_classifier.keras"
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


def _validated_inputs(payload: dict[str, Any]) -> tuple[np.ndarray, str]:
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
    return normalized[np.newaxis, ...], mode


def predict_state(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    model_input, input_mode = _validated_inputs(payload)
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
        "model_id": "friend-se-resnet-bidirectional-gru-attention",
        "model_output_type": "five-class telemetry-state pattern",
        "input_mode": input_mode,
        "top_class_index": top_index,
        "top_class_label": MODEL_LABELS[top_index],
        "top_class_raw_score": round(float(scores[top_index]), 6),
        "scores_are_calibrated_probabilities": False,
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
