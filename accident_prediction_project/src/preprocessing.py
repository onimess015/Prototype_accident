from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import standardize_columns


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    missing_markers = {"", "na", "n/a", "none", "null", "nan"}

    for column in converted.select_dtypes(include=["object", "string"]).columns:
        series = converted[column].astype("string").str.strip()
        series = series.replace(list(missing_markers), pd.NA)

        numeric_candidate = pd.to_numeric(series, errors="coerce")
        non_missing = series.notna().sum()
        convertible_ratio = 0.0
        if non_missing > 0:
            convertible_ratio = numeric_candidate.notna().sum() / non_missing

        if convertible_ratio >= 0.8:
            converted[column] = pd.Series(numeric_candidate, index=series.index).astype(
                float
            )
        else:
            object_series = series.astype(object)
            object_series[pd.isna(object_series)] = np.nan
            converted[column] = object_series

    return converted


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_columns(df)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = _coerce_object_columns(cleaned)
    return cleaned


def identify_column_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_columns = [
        column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns, categorical_columns = identify_column_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", _build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    preprocessor.feature_schema_ = list(X.columns)
    preprocessor.numeric_features_ = numeric_columns
    preprocessor.categorical_features_ = categorical_columns
    return preprocessor


def fit_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame):
    transformed_frame = X.copy()
    feature_schema = getattr(
        preprocessor, "feature_schema_", list(transformed_frame.columns)
    )

    for column in feature_schema:
        if column not in transformed_frame.columns:
            transformed_frame[column] = np.nan

    transformed_frame = transformed_frame[feature_schema]
    return preprocessor.transform(transformed_frame)
