from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


PREFERRED_FEATURES = [
    "driver_age",
    "driver_gender",
    "alcohol_involvement",
    "vehicle_type",
    "vehicle_speed",
    "road_type",
    "road_condition",
    "lighting_conditions",
    "weather_conditions",
    "traffic_control_presence",
    "time_of_day",
    "day_of_week",
]

FEATURE_ALIASES: dict[str, list[str]] = {
    "driver_age": ["driver_age", "age_of_driver"],
    "driver_gender": ["driver_gender", "gender_of_driver"],
    "alcohol_involvement": ["alcohol_involvement", "alcohol", "drunk_driving"],
    "vehicle_type": ["vehicle_type", "vehicle_type_involved", "type_of_vehicle"],
    "vehicle_speed": [
        "vehicle_speed",
        "speed_limit",
        "speed_limit_km_h",
        "speed_limit_km_h_",
    ],
    "road_type": ["road_type", "road_category"],
    "road_condition": ["road_condition", "surface_condition"],
    "lighting_conditions": ["lighting_conditions", "light_condition"],
    "weather_conditions": ["weather_conditions", "weather"],
    "traffic_control_presence": [
        "traffic_control_presence",
        "traffic_control",
        "traffic_signal_presence",
    ],
    "time_of_day": ["time_of_day", "time", "accident_time"],
    "day_of_week": ["day_of_week", "weekday"],
}

TARGET_ALIASES = [
    "accident_severity",
    "severity",
    "severity_level",
    "accident_risk",
]

LOW_RISK_TOKENS = {"low", "minor", "slight", "non_fatal", "nonfatal"}
HIGH_RISK_TOKENS = {"serious", "severe", "fatal", "high", "critical", "major"}


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    alias_set = {alias.lower() for alias in aliases}
    for column in columns:
        if column.lower() in alias_set:
            return column
    for alias in aliases:
        for column in columns:
            if alias.lower() in column.lower():
                return column
    return None


def _resolve_feature_mapping(df: pd.DataFrame) -> Dict[str, str]:
    available_columns = list(df.columns)
    feature_mapping: Dict[str, str] = {}

    for feature_name in PREFERRED_FEATURES:
        matched_column = _find_column(available_columns, FEATURE_ALIASES[feature_name])
        if matched_column:
            feature_mapping[feature_name] = matched_column

    return feature_mapping


def _resolve_target_column(df: pd.DataFrame) -> str:
    target_column = _find_column(list(df.columns), TARGET_ALIASES)
    if target_column:
        return target_column

    fallback_candidates = ["number_of_fatalities", "number_of_casualties"]
    fallback = _find_column(list(df.columns), fallback_candidates)
    if fallback:
        return fallback

    raise ValueError(
        "Unable to identify a target column. Expected accident severity or a close alternative."
    )


def create_binary_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        series = pd.to_numeric(y, errors="coerce")
        if series.isna().all():
            raise ValueError("Target column is numeric but could not be parsed.")
        return (series.fillna(series.median()) > 0).astype(int)

    normalized = (
        y.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    binary_target = pd.Series(np.nan, index=y.index, dtype="float")

    binary_target.loc[normalized.isin(LOW_RISK_TOKENS)] = 0
    binary_target.loc[normalized.isin(HIGH_RISK_TOKENS)] = 1

    remaining_mask = binary_target.isna()
    if remaining_mask.any():
        binary_target.loc[remaining_mask] = normalized.loc[remaining_mask].apply(
            lambda value: 1 if any(token in value for token in HIGH_RISK_TOKENS) else 0
        )

    return binary_target.astype(int)


def select_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_mapping = _resolve_feature_mapping(df)
    if len(feature_mapping) < 4:
        raise ValueError(
            "Insufficient usable features found in the training dataset. "
            f"Detected features: {sorted(feature_mapping)}"
        )

    target_column = _resolve_target_column(df)

    X = pd.DataFrame(index=df.index)
    for feature_name in PREFERRED_FEATURES:
        if feature_name in feature_mapping:
            X[feature_name] = df[feature_mapping[feature_name]]

    y = create_binary_target(df[target_column])
    valid_rows = y.notna()
    return X.loc[valid_rows].reset_index(drop=True), y.loc[valid_rows].astype(
        int
    ).reset_index(drop=True)


def get_feature_schema() -> list[str]:
    return PREFERRED_FEATURES.copy()
