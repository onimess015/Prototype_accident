from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import load_main_dataset
from .utils import standardize_columns


REQUIRED_MODEL_FEATURES = [
    "vehicle_speed",
    "weather_condition",
    "road_type",
    "traffic_density",
    "time_of_day",
    "driver_fatigue",
    "road_lighting",
    "visibility_level",
    "driver_age",
    "driver_gender",
    "alcohol_involvement",
    "vehicle_type",
    "road_condition",
    "traffic_control_presence",
    "day_of_week",
    "month",
]


def _extract_hour(value: Any) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        # Handles formats like 23:10 or 7:5
        return int(text.split(":", maxsplit=1)[0])
    except (ValueError, TypeError):
        return None


def _to_day_night(hour: int | None) -> str:
    if hour is None:
        return "Unknown"
    return "Day" if 6 <= hour < 18 else "Night"


def _derive_traffic_density(road_type: str, hour: int | None) -> str:
    road = str(road_type).lower()
    if hour is None:
        return "Medium"

    peak_hour = (7 <= hour <= 10) or (17 <= hour <= 21)
    off_peak = hour < 6 or hour >= 23

    if "urban" in road or "city" in road:
        if peak_hour:
            return "High"
        if off_peak:
            return "Low"
        return "Medium"

    if "highway" in road:
        if peak_hour:
            return "Medium"
        return "Low"

    if "rural" in road or "village" in road:
        if peak_hour:
            return "Medium"
        return "Low"

    return "Medium"


def _derive_driver_fatigue(hour: int | None) -> str:
    if hour is None:
        return "Unknown"

    if 0 <= hour <= 5:
        return "High"
    if 13 <= hour <= 16 or 22 <= hour <= 23:
        return "Medium"
    return "Low"


def _derive_visibility(weather: str, lighting: str) -> str:
    weather_text = str(weather).lower()
    lighting_text = str(lighting).lower()

    if any(token in weather_text for token in ["fog", "storm", "rain", "snow"]):
        if "dark" in lighting_text or "dusk" in lighting_text:
            return "Low"
        return "Medium"

    if "dark" in lighting_text:
        return "Medium"

    return "High"


def _build_target(series: pd.Series) -> pd.Series:
    # Severe outcomes are positive class for risk modeling.
    mapping = {
        "minor": 0,
        "slight": 0,
        "low": 0,
        "serious": 1,
        "severe": 1,
        "fatal": 1,
        "high": 1,
        "critical": 1,
    }

    normalized = series.astype("string").str.strip().str.lower().fillna("unknown")
    encoded = normalized.map(mapping)
    encoded = encoded.fillna(
        normalized.str.contains("fatal|severe|serious").astype(int)
    )
    return encoded.astype(int)


def _clean_target_consistency(
    severity_target: pd.Series,
    fatalities: pd.Series,
    casualties: pd.Series,
) -> pd.Series:
    """Resolve obvious label contradictions using casualty metadata."""
    cleaned = severity_target.copy()

    fatal_count = pd.to_numeric(fatalities, errors="coerce").fillna(0)
    casualty_count = pd.to_numeric(casualties, errors="coerce").fillna(0)

    # Only extreme casualty metadata overrides explicit severity labels.
    cleaned.loc[(cleaned == 0) & (fatal_count >= 3)] = 1
    cleaned.loc[(cleaned == 0) & (casualty_count >= 10)] = 1

    return cleaned.astype(int)


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_holdout: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_holdout: pd.Series
    quality_report: dict[str, Any]


def _canonicalize_binary_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.replace(
        {
            "true": "yes",
            "false": "no",
            "1": "yes",
            "0": "no",
            "y": "yes",
            "n": "no",
        }
    )


def _run_data_quality_filters(
    df: pd.DataFrame,
    source_speed: str,
    source_driver_age: str,
    source_target: str,
    source_fatalities: str,
    source_casualties: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "rows_before": int(len(df)),
        "dropped_rows": 0,
        "drop_reasons": {},
    }

    filtered = df.copy()

    # Must have at least one of the core fields present.
    missing_core_mask = filtered[source_target].isna() & filtered[source_speed].isna()
    if missing_core_mask.any():
        dropped = int(missing_core_mask.sum())
        filtered = filtered.loc[~missing_core_mask].copy()
        report["drop_reasons"]["missing_target_and_speed"] = dropped

    speed = pd.to_numeric(filtered[source_speed], errors="coerce")
    invalid_speed_mask = speed > 220
    if invalid_speed_mask.any():
        dropped = int(invalid_speed_mask.sum())
        filtered = filtered.loc[~invalid_speed_mask].copy()
        report["drop_reasons"]["unrealistic_speed_over_220"] = dropped

    age = pd.to_numeric(filtered[source_driver_age], errors="coerce")
    invalid_age_mask = (age < 14) | (age > 100)
    if invalid_age_mask.any():
        filtered.loc[invalid_age_mask, source_driver_age] = np.nan
        report["drop_reasons"]["driver_age_set_missing"] = int(invalid_age_mask.sum())

    if "alcohol_involvement" in filtered.columns:
        filtered["alcohol_involvement"] = _canonicalize_binary_text(
            filtered["alcohol_involvement"]
        )

    report["rows_after"] = int(len(filtered))
    report["dropped_rows"] = int(report["rows_before"] - report["rows_after"])

    target_preview = _build_target(filtered[source_target])
    target_preview = _clean_target_consistency(
        target_preview,
        fatalities=filtered[source_fatalities],
        casualties=filtered[source_casualties],
    )
    report["target_distribution_after"] = {
        str(k): int(v) for k, v in target_preview.value_counts().sort_index().items()
    }

    return filtered, report


def load_and_prepare_data(
    test_size: float = 0.2,
    holdout_size: float = 0.1,
    random_state: int = 42,
) -> PreparedData:
    df = standardize_columns(load_main_dataset())

    # Resolve source columns with robust fallbacks.
    source_speed = (
        "speed_limit_km_h" if "speed_limit_km_h" in df.columns else "speed_limit"
    )
    source_weather = (
        "weather_conditions" if "weather_conditions" in df.columns else "weather"
    )
    source_road = "road_type" if "road_type" in df.columns else "road_category"
    source_time = "time_of_day" if "time_of_day" in df.columns else "time"
    source_lighting = (
        "lighting_conditions"
        if "lighting_conditions" in df.columns
        else "light_condition"
    )
    source_target = (
        "accident_severity" if "accident_severity" in df.columns else "severity"
    )
    source_driver_age = "driver_age" if "driver_age" in df.columns else "age_of_driver"
    source_driver_gender = (
        "driver_gender" if "driver_gender" in df.columns else "gender_of_driver"
    )
    source_vehicle_type = (
        "vehicle_type_involved"
        if "vehicle_type_involved" in df.columns
        else "vehicle_type"
    )
    source_road_condition = (
        "road_condition" if "road_condition" in df.columns else "surface_condition"
    )
    source_traffic_control = (
        "traffic_control_presence"
        if "traffic_control_presence" in df.columns
        else "traffic_control"
    )
    source_day_of_week = "day_of_week" if "day_of_week" in df.columns else "weekday"
    source_month = "month" if "month" in df.columns else "accident_month"
    source_fatalities = (
        "number_of_fatalities" if "number_of_fatalities" in df.columns else "fatalities"
    )
    source_casualties = (
        "number_of_casualties" if "number_of_casualties" in df.columns else "casualties"
    )

    for required_source in [
        source_speed,
        source_target,
        source_driver_age,
        source_fatalities,
        source_casualties,
    ]:
        if required_source not in df.columns:
            df[required_source] = np.nan

    df, quality_report = _run_data_quality_filters(
        df=df,
        source_speed=source_speed,
        source_driver_age=source_driver_age,
        source_target=source_target,
        source_fatalities=source_fatalities,
        source_casualties=source_casualties,
    )

    engineered = pd.DataFrame(index=df.index)
    engineered["vehicle_speed"] = pd.to_numeric(df.get(source_speed), errors="coerce")
    engineered["weather_condition"] = df.get(source_weather, "Unknown").astype("string")
    engineered["road_type"] = df.get(source_road, "Unknown").astype("string")
    engineered["road_lighting"] = df.get(source_lighting, "Unknown").astype("string")
    engineered["driver_age"] = pd.to_numeric(df.get(source_driver_age), errors="coerce")
    engineered["driver_gender"] = df.get(source_driver_gender, "Unknown").astype(
        "string"
    )
    engineered["alcohol_involvement"] = df.get("alcohol_involvement", "Unknown").astype(
        "string"
    )
    engineered["vehicle_type"] = df.get(source_vehicle_type, "Unknown").astype("string")
    engineered["road_condition"] = df.get(source_road_condition, "Unknown").astype(
        "string"
    )
    engineered["traffic_control_presence"] = df.get(
        source_traffic_control, "Unknown"
    ).astype("string")
    engineered["day_of_week"] = df.get(source_day_of_week, "Unknown").astype("string")
    engineered["month"] = df.get(source_month, "Unknown").astype("string")

    hours = df.get(source_time, pd.Series([None] * len(df))).map(_extract_hour)
    engineered["time_of_day"] = hours.map(_to_day_night)
    engineered["traffic_density"] = [
        _derive_traffic_density(road_type, hour)
        for road_type, hour in zip(engineered["road_type"], hours)
    ]
    engineered["driver_fatigue"] = hours.map(_derive_driver_fatigue)
    engineered["visibility_level"] = [
        _derive_visibility(weather, lighting)
        for weather, lighting in zip(
            engineered["weather_condition"], engineered["road_lighting"]
        )
    ]

    # Basic numeric cleanup and clipping to realistic values.
    engineered["vehicle_speed"] = engineered["vehicle_speed"].fillna(
        engineered["vehicle_speed"].median()
    )
    engineered["vehicle_speed"] = engineered["vehicle_speed"].clip(lower=0, upper=150)
    engineered["driver_age"] = engineered["driver_age"].fillna(
        engineered["driver_age"].median()
    )
    engineered["driver_age"] = engineered["driver_age"].clip(lower=16, upper=90)

    categorical_features = [
        feature
        for feature in REQUIRED_MODEL_FEATURES
        if feature not in {"vehicle_speed", "driver_age"}
    ]
    for feature in categorical_features:
        categorical_series = engineered[feature].astype(object)
        categorical_series[pd.isna(categorical_series)] = np.nan
        engineered[feature] = categorical_series

    severity_target = _build_target(
        df.get(source_target, pd.Series(["minor"] * len(df), index=df.index))
    )
    target = _clean_target_consistency(
        severity_target=severity_target,
        fatalities=df.get(source_fatalities, pd.Series([0] * len(df), index=df.index)),
        casualties=df.get(source_casualties, pd.Series([0] * len(df), index=df.index)),
    )

    X = engineered[REQUIRED_MODEL_FEATURES].copy()
    y = target

    stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    stratify_dev = (
        y_dev if y_dev.nunique() > 1 and y_dev.value_counts().min() >= 2 else None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_dev,
        y_dev,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_dev,
    )

    quality_report["split_sizes"] = {
        "train": int(len(X_train)),
        "test": int(len(X_test)),
        "holdout": int(len(X_holdout)),
    }

    return PreparedData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        X_holdout=X_holdout.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_holdout=y_holdout.reset_index(drop=True),
        quality_report=quality_report,
    )
