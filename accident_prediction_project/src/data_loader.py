from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import DATA_DIR, resolve_existing_path, safe_read_table, standardize_columns


def _load_dataset(
    file_candidates: list[str], data_dir: Path | None = None
) -> pd.DataFrame:
    file_path = resolve_existing_path(file_candidates, base_dir=data_dir or DATA_DIR)
    dataframe = safe_read_table(file_path)
    return standardize_columns(dataframe)


def load_main_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_dataset(
        [
            "accident_prediction_india.xls",
            "accident_prediction_india.xlsx",
            "accident_prediction_india.csv",
        ],
        data_dir=data_dir,
    )


def load_cause_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_dataset(
        [
            "cause-wise-distribution-of-road-accidents-and-unmanned-railway-crossing-accidents.csv",
            "cause-wise-distribution-of-railway-accidents.csv",
        ],
        data_dir=data_dir,
    )


def load_road_class_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_dataset(
        ["road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv"],
        data_dir=data_dir,
    )


def load_time_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_dataset(
        ["time-of-occurrence-wise-number-of-traffic-accidents.csv"],
        data_dir=data_dir,
    )


def load_month_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    return _load_dataset(
        ["month-of-occurrence-wise-number-of-traffic-accidents.csv"],
        data_dir=data_dir,
    )
