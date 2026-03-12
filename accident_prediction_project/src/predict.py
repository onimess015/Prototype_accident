from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import get_feature_schema
from .preprocessing import transform_features
from .utils import MODELS_DIR


def load_artifacts(
    model_path: Path | None = None, preprocessor_path: Path | None = None
):
    resolved_model_path = model_path or (MODELS_DIR / "accident_model.pkl")
    resolved_preprocessor_path = preprocessor_path or (MODELS_DIR / "preprocessor.pkl")

    if not resolved_model_path.exists() or not resolved_preprocessor_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m src.train_model` from the project root first."
        )

    model = joblib.load(resolved_model_path)
    preprocessor = joblib.load(resolved_preprocessor_path)
    return model, preprocessor


def prepare_single_input(input_dict: dict, feature_schema: list[str]) -> pd.DataFrame:
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


def predict_risk(input_dict: dict) -> dict[str, float | str]:
    model, preprocessor = load_artifacts()
    feature_schema = list(
        getattr(preprocessor, "feature_schema_", get_feature_schema())
    )
    input_frame = prepare_single_input(input_dict, feature_schema)
    transformed_input = transform_features(preprocessor, input_frame)

    if hasattr(model, "predict_proba"):
        risk_score = float(model.predict_proba(transformed_input)[0][1])
    else:
        risk_score = float(model.predict(transformed_input)[0])

    risk_label = _score_to_label(risk_score)
    return {
        "risk_score": round(risk_score, 4),
        "risk_percentage": round(risk_score * 100, 2),
        "risk_label": risk_label,
    }
