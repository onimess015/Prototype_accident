from __future__ import annotations

import pandas as pd
import plotly.express as px

from .data_loader import (
    load_cause_dataset,
    load_month_dataset,
    load_road_class_dataset,
    load_time_dataset,
)
from .utils import MONTH_ORDER, TIME_BUCKET_ORDER


def _filter_road_accidents(df: pd.DataFrame) -> pd.DataFrame:
    if "accident_type" not in df.columns:
        return df.copy()

    mask = (
        df["accident_type"]
        .astype("string")
        .str.contains("road accident", case=False, na=False)
    )
    filtered = df.loc[mask].copy()
    return filtered if not filtered.empty else df.copy()


def plot_accidents_by_cause(df: pd.DataFrame | None = None):
    source = load_cause_dataset() if df is None else df.copy()
    grouped = (
        source.groupby("cause", as_index=False)["cases"]
        .sum()
        .sort_values("cases", ascending=False)
        .head(10)
    )
    figure = px.bar(
        grouped,
        x="cases",
        y="cause",
        orientation="h",
        color="cases",
        color_continuous_scale="Tealgrn",
        title="Top Accident Causes by Number of Cases",
        labels={"cases": "Accident Cases", "cause": "Cause"},
    )
    figure.update_layout(
        yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False
    )
    return figure, grouped


def plot_accidents_by_road_type(df: pd.DataFrame | None = None):
    source = load_road_class_dataset() if df is None else df.copy()
    grouped = (
        source.groupby("road_type", as_index=False)["cases"]
        .sum()
        .sort_values("cases", ascending=False)
    )
    figure = px.bar(
        grouped,
        x="road_type",
        y="cases",
        color="road_type",
        title="Accident Cases by Road Type",
        labels={"road_type": "Road Type", "cases": "Accident Cases"},
    )
    figure.update_layout(showlegend=False)
    return figure, grouped


def plot_accidents_by_time(df: pd.DataFrame | None = None):
    source = _filter_road_accidents(load_time_dataset() if df is None else df.copy())
    grouped = source.groupby("time", as_index=False)["number_of_accidents"].sum()
    grouped["time"] = pd.Categorical(
        grouped["time"], categories=TIME_BUCKET_ORDER, ordered=True
    )
    grouped = grouped.sort_values("time")
    figure = px.line(
        grouped,
        x="time",
        y="number_of_accidents",
        markers=True,
        title="Road Accidents by Time of Day",
        labels={"time": "Time of Day", "number_of_accidents": "Number of Accidents"},
    )
    return figure, grouped


def plot_accidents_by_month(df: pd.DataFrame | None = None):
    source = _filter_road_accidents(load_month_dataset() if df is None else df.copy())
    grouped = source.groupby("month", as_index=False)["number_of_accidents"].sum()
    grouped["month"] = pd.Categorical(
        grouped["month"], categories=MONTH_ORDER, ordered=True
    )
    grouped = grouped.sort_values("month")
    figure = px.area(
        grouped,
        x="month",
        y="number_of_accidents",
        title="Monthly Road Accident Trend",
        labels={"month": "Month", "number_of_accidents": "Number of Accidents"},
    )
    return figure, grouped
