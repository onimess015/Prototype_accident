from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import joblib
import numpy as np
import pandas as pd

from .data_preprocessing import REQUIRED_MODEL_FEATURES, NUMERIC_MODEL_FEATURES
from .utils import MODELS_DIR


def load_artifacts(
    model_path: Path | None = None, preprocessor_path: Path | None = None
):
    resolved_model_path = model_path or (MODELS_DIR / "model.pkl")
    fallback_model_path = MODELS_DIR / "accident_model.pkl"
    resolved_preprocessor_path = preprocessor_path or (MODELS_DIR / "preprocessor.pkl")

    if not resolved_model_path.exists() and fallback_model_path.exists():
        resolved_model_path = fallback_model_path

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m src.train_model` from the project root first."
        )

    model = joblib.load(resolved_model_path)
    preprocessor = None
    if resolved_preprocessor_path.exists():
        preprocessor = joblib.load(resolved_preprocessor_path)

    return model, preprocessor


def _validate_payload(input_dict: dict[str, Any]) -> dict[str, Any]:
    validated = dict(input_dict)

    alias_map = {
        "weather_conditions": "weather_condition",
        "lighting_conditions": "road_lighting",
        "traffic_control": "traffic_control_presence",
    }
    for alias_name, canonical_name in alias_map.items():
        if canonical_name not in validated and alias_name in validated:
            validated[canonical_name] = validated[alias_name]

    speed_value = pd.to_numeric(validated.get("vehicle_speed"), errors="coerce")
    if pd.isna(speed_value):
        raise ValueError("Vehicle speed is required and must be numeric.")
    if speed_value < 0:
        raise ValueError("Vehicle speed cannot be negative.")
    if speed_value > 150:
        raise ValueError(
            "Vehicle speed is unrealistically high. Please use 0-150 km/h."
        )

    validated["vehicle_speed"] = float(speed_value)

    for feature_name in REQUIRED_MODEL_FEATURES:
        if feature_name not in validated:
            validated[feature_name] = (
                np.nan if feature_name in NUMERIC_MODEL_FEATURES else "Unknown"
            )

    for boolean_like_feature in ["alcohol_involvement"]:
        raw_value = str(validated.get(boolean_like_feature, "Unknown")).strip().lower()
        if raw_value in {"1", "true", "yes", "y"}:
            validated[boolean_like_feature] = "Yes"
        elif raw_value in {"0", "false", "no", "n"}:
            validated[boolean_like_feature] = "No"

    weather = str(validated.get("weather_condition", "Unknown"))
    lighting = str(validated.get("road_lighting", "Unknown"))
    traffic = str(validated.get("traffic_density", "Unknown"))
    time_of_day = str(validated.get("time_of_day", "Unknown"))

    speed = float(validated.get("vehicle_speed", 0.0) or 0.0)
    if speed < 40:
        validated["speed_band"] = "low"
    elif speed < 70:
        validated["speed_band"] = "moderate"
    elif speed < 100:
        validated["speed_band"] = "high"
    else:
        validated["speed_band"] = "very_high"

    weather_text = weather.strip().lower()
    if weather_text in {"clear", "sunny"}:
        validated["weather_severity"] = "benign"
    elif weather_text in {"hazy", "cloudy"}:
        validated["weather_severity"] = "mild"
    elif weather_text in {"rain", "rainy", "fog", "foggy", "snow"}:
        validated["weather_severity"] = "adverse"
    elif weather_text in {"storm", "stormy", "thunderstorm"}:
        validated["weather_severity"] = "severe"
    else:
        validated["weather_severity"] = "unknown"

    time_text = time_of_day.strip().lower()
    traffic_text = traffic.strip().lower()
    light_text = lighting.strip().lower()
    if time_text == "night" and (
        traffic_text == "high" or light_text in {"dark", "poor", "dusk", "dawn"}
    ):
        validated["contextual_risk"] = "elevated"
    elif traffic_text == "high":
        validated["contextual_risk"] = "moderate"
    else:
        validated["contextual_risk"] = "baseline"

    return validated


def prepare_single_input(
    input_dict: dict[str, Any], feature_schema: list[str]
) -> pd.DataFrame:
    record = {}
    for feature_name in feature_schema:
        value = input_dict.get(feature_name, np.nan)
        record[feature_name] = np.nan if value in (None, "") else value
    return pd.DataFrame([record], columns=feature_schema)


def _score_to_label(risk_score: float) -> str:
    if risk_score < 0.40:
        return "Low Risk"
    if risk_score < 0.70:
        return "Medium Risk"
    return "High Risk"


def _scenario_signature(input_dict: dict[str, Any]) -> str:
    keys = sorted(REQUIRED_MODEL_FEATURES)
    signature_base = "|".join(f"{key}={input_dict.get(key, 'unknown')}" for key in keys)
    return hashlib.sha1(signature_base.encode("utf-8")).hexdigest()[:10]


def _build_risk_narrative(
    input_dict: dict[str, Any],
    risk_score: float,
    risk_label: str,
) -> dict[str, str]:
    speed = float(
        pd.to_numeric(input_dict.get("vehicle_speed"), errors="coerce") or 0.0
    )
    weather = str(input_dict.get("weather_condition", "Unknown")).strip()
    road_type = str(input_dict.get("road_type", "Unknown")).strip()
    traffic = str(input_dict.get("traffic_density", "Unknown")).strip()
    time_of_day = str(input_dict.get("time_of_day", "Unknown")).strip()
    fatigue = str(input_dict.get("driver_fatigue", "Unknown")).strip()
    lighting = str(input_dict.get("road_lighting", "Unknown")).strip()
    road_condition = str(input_dict.get("road_condition", "Unknown")).strip()

    causes: list[str] = []
    how_steps: list[str] = []

    if speed >= 90:
        causes.append("high speed")
        how_steps.append(
            "Higher speed shortens reaction time and increases stopping distance"
        )
    elif speed >= 70:
        causes.append("moderate-high speed")
        how_steps.append("Speed leaves less margin for sudden hazards")

    if weather.lower() in {"fog", "foggy", "rain", "rainy", "snow", "stormy", "storm"}:
        causes.append(f"adverse weather ({weather})")
        how_steps.append(
            "Weather reduces traction or visibility, making control harder"
        )

    if lighting.lower() in {"dark", "dusk", "dawn", "poor"}:
        causes.append(f"low-light driving ({lighting})")
        how_steps.append("Low light delays hazard detection and depth judgment")

    if traffic.lower() in {"high", "dense", "heavy"}:
        causes.append("dense traffic")
        how_steps.append(
            "Dense traffic increases conflict points and sudden braking events"
        )

    if fatigue.lower() in {"high", "medium"}:
        causes.append(f"driver fatigue ({fatigue})")
        how_steps.append("Fatigue reduces attention consistency and reaction quality")

    if road_condition.lower() in {"wet", "damaged", "under construction"}:
        causes.append(f"road condition ({road_condition})")
        how_steps.append("Road surface condition can reduce grip and stability")

    if not causes:
        causes.append("baseline environmental uncertainty")
        how_steps.append(
            "Combined road context still carries non-zero incident probability"
        )

    reason = (
        f"Predicted {risk_label.lower()} because this scenario combines "
        f"{', '.join(causes[:3])}."
    )
    cause = f"Primary likely cause: {causes[0]}."
    why = (
        f"Why: the model score is {risk_score:.2f}, driven by current road context "
        f"(road={road_type}, time={time_of_day}, traffic={traffic})."
    )
    how = f"How accident can happen: {'; '.join(how_steps[:3])}."

    return {
        "scenario_signature": _scenario_signature(input_dict),
        "accident_reason": reason,
        "accident_cause": cause,
        "accident_why": why,
        "accident_how": how,
    }


def _normalize_importance_name(raw_name: str) -> str:
    if "__" in raw_name:
        raw_name = raw_name.split("__", maxsplit=1)[1]

    for feature in REQUIRED_MODEL_FEATURES:
        if raw_name.startswith(feature):
            return feature
    return raw_name


def _top_risk_factors(model, input_frame: pd.DataFrame) -> list[str]:
    if hasattr(model, "top_features_") and model.top_features_:
        return [str(name).replace("_", " ").title() for name in model.top_features_[:3]]

    if not hasattr(model, "named_steps"):
        return []

    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")
    if classifier is None or preprocessor is None:
        return []

    if hasattr(classifier, "feature_importances_"):
        raw_importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        raw_importance = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        return []

    transformed_names = preprocessor.get_feature_names_out()
    mapped = pd.DataFrame(
        {
            "feature": [_normalize_importance_name(name) for name in transformed_names],
            "importance": raw_importance,
        }
    )
    grouped = (
        mapped.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    top_names = grouped.head(3)["feature"].tolist()
    readable = [name.replace("_", " ").title() for name in top_names]
    return readable


def predict_risk(input_dict: dict[str, Any]) -> dict[str, float | str | list[str]]:
    validated_input = _validate_payload(input_dict)
    model, preprocessor = load_artifacts()

    if hasattr(model, "feature_schema_"):
        feature_schema = list(model.feature_schema_)
    elif preprocessor is not None and hasattr(preprocessor, "feature_schema_"):
        feature_schema = list(preprocessor.feature_schema_)
    else:
        feature_schema = REQUIRED_MODEL_FEATURES.copy()

    input_frame = prepare_single_input(validated_input, feature_schema)

    if hasattr(model, "predict_proba"):
        risk_score = float(model.predict_proba(input_frame)[0][1])
    else:
        risk_score = float(model.predict(input_frame)[0])

    decision_threshold = float(getattr(model, "best_threshold_", 0.5))
    predicted_class = int(risk_score >= decision_threshold)

    risk_label = _score_to_label(risk_score)
    top_factors = _top_risk_factors(model, input_frame)
    narrative = _build_risk_narrative(validated_input, risk_score, risk_label)
    return {
        "risk_score": round(risk_score, 4),
        "risk_percentage": round(risk_score * 100, 2),
        "risk_label": risk_label,
        "predicted_class": predicted_class,
        "decision_threshold": round(decision_threshold, 3),
        "top_factors": top_factors,
        **narrative,
    }
