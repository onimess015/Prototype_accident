from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_and_prepare_data(
    test_size: float = 0.2, random_state: int = 42
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

    engineered = pd.DataFrame(index=df.index)
    engineered["vehicle_speed"] = pd.to_numeric(df.get(source_speed), errors="coerce")
    engineered["weather_condition"] = df.get(source_weather, "Unknown").astype("string")
    engineered["road_type"] = df.get(source_road, "Unknown").astype("string")
    engineered["road_lighting"] = df.get(source_lighting, "Unknown").astype("string")

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

    target = _build_target(df.get(source_target, pd.Series(["minor"] * len(df))))

    X = engineered[REQUIRED_MODEL_FEATURES].copy()
    y = target

    stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return PreparedData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
