from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_preprocessing import REQUIRED_MODEL_FEATURES
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
            validated[feature_name] = "Unknown"

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


def _normalize_importance_name(raw_name: str) -> str:
    if "__" in raw_name:
        raw_name = raw_name.split("__", maxsplit=1)[1]

    for feature in REQUIRED_MODEL_FEATURES:
        if raw_name.startswith(feature):
            return feature
    return raw_name


def _top_risk_factors(model, input_frame: pd.DataFrame) -> list[str]:
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

    risk_label = _score_to_label(risk_score)
    top_factors = _top_risk_factors(model, input_frame)
    return {
        "risk_score": round(risk_score, 4),
        "risk_percentage": round(risk_score * 100, 2),
        "risk_label": risk_label,
        "top_factors": top_factors,
    }
