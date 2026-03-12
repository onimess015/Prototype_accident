from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

TIME_BUCKET_ORDER = [
    "12 AM to 3 AM",
    "3 AM to 6 AM",
    "6 AM to 9 AM",
    "9 AM to 12 PM",
    "12 PM to 3 PM",
    "3 PM to 6 PM",
    "6 PM to 9 PM",
    "9 PM to 12 AM",
]


def standardize_column_name(name: str) -> str:
    normalized = str(name).strip().lower().replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [standardize_column_name(column) for column in cleaned.columns]
    unnamed_columns = [
        column for column in cleaned.columns if column.startswith("unnamed")
    ]
    if unnamed_columns:
        cleaned = cleaned.drop(columns=unnamed_columns)
    return cleaned


def resolve_existing_path(
    file_candidates: Iterable[str], base_dir: Path | None = None
) -> Path:
    search_dir = base_dir or DATA_DIR
    for candidate in file_candidates:
        candidate_path = search_dir / candidate
        if candidate_path.exists():
            return candidate_path
    searched = ", ".join(str(search_dir / candidate) for candidate in file_candidates)
    raise FileNotFoundError(f"None of the expected files were found: {searched}")


def safe_read_table(file_path: Path) -> pd.DataFrame:
    try:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        if file_path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(file_path)
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Dataset file not found: {file_path}") from error
    except Exception as error:  # pragma: no cover - defensive path
        raise RuntimeError(f"Failed to load dataset: {file_path}. {error}") from error


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_feature_label(feature_name: str) -> str:
    custom_labels = {
        "vehicle_speed": "Vehicle Speed / Speed Limit (km/h)",
        "time_of_day": "Time of Day",
        "day_of_week": "Day of Week",
        "driver_age": "Driver Age",
        "driver_gender": "Driver Gender",
        "road_type": "Road Type",
        "road_condition": "Road Condition",
        "lighting_conditions": "Lighting Conditions",
        "weather_conditions": "Weather Conditions",
        "traffic_control_presence": "Traffic Control Presence",
        "vehicle_type": "Vehicle Type",
        "alcohol_involvement": "Alcohol Involvement",
    }
    return custom_labels.get(feature_name, feature_name.replace("_", " ").title())
