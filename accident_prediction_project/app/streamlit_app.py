from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (  # noqa: E402
    plot_accidents_by_cause,
    plot_accidents_by_month,
    plot_accidents_by_road_type,
    plot_accidents_by_time,
)
from src.data_loader import load_main_dataset  # noqa: E402
from src.feature_engineering import (
    get_feature_schema,
    select_features_and_target,
)  # noqa: E402
from src.predict import load_artifacts, predict_risk  # noqa: E402
from src.preprocessing import clean_dataframe  # noqa: E402
from src.utils import format_feature_label  # noqa: E402
from prediction_ui import RiskPredictionUI  # noqa: E402


st.set_page_config(page_title="SafeRoute AI", page_icon="🚗", layout="wide")


@st.cache_data(show_spinner=False)
def load_feature_reference_data() -> pd.DataFrame:
    dataframe = clean_dataframe(load_main_dataset())
    features, _ = select_features_and_target(dataframe)
    return features


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    return load_artifacts()


def _build_categorical_options(
    feature_frame: pd.DataFrame, feature_name: str
) -> list[str]:
    if feature_name not in feature_frame.columns:
        return ["Unknown"]
    values = (
        feature_frame[feature_name]
        .dropna()
        .astype("string")
        .replace("<NA>", pd.NA)
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    return values or ["Unknown"]


def _build_numeric_defaults(
    feature_frame: pd.DataFrame, feature_name: str, default_value: int
) -> tuple[int, int, int]:
    if feature_name not in feature_frame.columns:
        return 0, 120, default_value

    series = pd.to_numeric(feature_frame[feature_name], errors="coerce").dropna()
    if series.empty:
        return 0, 120, default_value

    minimum = int(series.min())
    maximum = int(series.max())
    median = int(series.median())
    return minimum, maximum, median


def _prediction_message(label: str) -> tuple[str, str]:
    if label == "Low Risk":
        return "#e7f7ee", "#18794e"
    if label == "Medium Risk":
        return "#fff4db", "#b26a00"
    return "#fdecea", "#b42318"


def render_prediction_form(feature_frame: pd.DataFrame) -> dict[str, object]:
    available_features = [
        feature for feature in get_feature_schema() if feature in feature_frame.columns
    ]
    input_payload: dict[str, object] = {}

    driver_column, vehicle_column = st.columns(2)
    road_column, environment_column = st.columns(2)

    with driver_column:
        st.subheader("Driver Information")
        if "driver_age" in available_features:
            minimum, maximum, median = _build_numeric_defaults(
                feature_frame, "driver_age", 35
            )
            input_payload["driver_age"] = st.slider(
                format_feature_label("driver_age"),
                min_value=minimum,
                max_value=max(maximum, minimum + 1),
                value=median,
            )
        if "driver_gender" in available_features:
            options = _build_categorical_options(feature_frame, "driver_gender")
            input_payload["driver_gender"] = st.selectbox(
                format_feature_label("driver_gender"), options
            )
        if "alcohol_involvement" in available_features:
            options = _build_categorical_options(feature_frame, "alcohol_involvement")
            input_payload["alcohol_involvement"] = st.selectbox(
                format_feature_label("alcohol_involvement"), options
            )
        if "day_of_week" in available_features:
            options = _build_categorical_options(feature_frame, "day_of_week")
            input_payload["day_of_week"] = st.selectbox(
                format_feature_label("day_of_week"), options
            )

    with vehicle_column:
        st.subheader("Vehicle Information")
        if "vehicle_type" in available_features:
            options = _build_categorical_options(feature_frame, "vehicle_type")
            input_payload["vehicle_type"] = st.selectbox(
                format_feature_label("vehicle_type"), options
            )
        if "vehicle_speed" in available_features:
            minimum, maximum, median = _build_numeric_defaults(
                feature_frame, "vehicle_speed", 60
            )
            input_payload["vehicle_speed"] = st.slider(
                format_feature_label("vehicle_speed"),
                min_value=minimum,
                max_value=max(maximum, minimum + 1),
                value=median,
            )

    with road_column:
        st.subheader("Road Conditions")
        for feature_name in [
            "road_type",
            "road_condition",
            "lighting_conditions",
            "traffic_control_presence",
        ]:
            if feature_name in available_features:
                options = _build_categorical_options(feature_frame, feature_name)
                input_payload[feature_name] = st.selectbox(
                    format_feature_label(feature_name), options
                )

    with environment_column:
        st.subheader("Environment")
        for feature_name in ["weather_conditions", "time_of_day"]:
            if feature_name in available_features:
                options = _build_categorical_options(feature_frame, feature_name)
                input_payload[feature_name] = st.selectbox(
                    format_feature_label(feature_name), options
                )

    return input_payload


st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f3d3e, #185d5f 55%, #d7f3e3 140%);
        color: #f7fbf8;
        margin-bottom: 1.25rem;
        box-shadow: 0 18px 40px rgba(15, 61, 62, 0.18);
    }
    .status-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        font-weight: 600;
        margin-top: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.35rem;">SafeRoute AI</h1>
        <p style="font-size:1.04rem; margin-bottom:0.3rem;">
            Road Accident Risk Prediction and Analytics Dashboard
        </p>
        <p style="margin:0; opacity:0.92;">
            Streamlit-based machine learning system for accident risk scoring and interactive road safety analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    feature_reference = load_feature_reference_data()
except Exception as error:
    st.error(f"Failed to load the training dataset: {error}")
    st.stop()

tabs = st.tabs(
    [
        "Accident Risk Prediction",
        "Accident Causes Analytics",
        "Road Type Analytics",
        "Time Trends",
        "Monthly Trends",
    ]
)

with tabs[0]:
    st.write(
        "Use the contextual inputs below to estimate the probability of a severe road accident under similar conditions."
    )
    user_input = render_prediction_form(feature_reference)
    prediction_button = st.button(
        "Predict Accident Risk", type="primary", use_container_width=True
    )

    if prediction_button:
        try:
            load_model_artifacts()
            prediction = predict_risk(user_input)

            # Use professional prediction UI
            RiskPredictionUI.render_prediction_result(
                probability=prediction["risk_score"],
                user_inputs=user_input,
                show_gauge=True,
            )
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Prediction failed: {error}")

with tabs[1]:
    st.subheader("Accident Cause Distribution")
    figure, summary = plot_accidents_by_cause()
    st.plotly_chart(figure, use_container_width=True)
    top_row = summary.iloc[0]
    st.write(
        f"The leading reported cause in the supporting dataset is {top_row['cause']} with {int(top_row['cases']):,} cases."
    )

with tabs[2]:
    st.subheader("Road Type Analysis")
    figure, summary = plot_accidents_by_road_type()
    st.plotly_chart(figure, use_container_width=True)
    top_row = summary.iloc[0]
    st.write(
        f"{top_row['road_type']} records the highest accident volume in the aggregated road classification dataset."
    )

with tabs[3]:
    st.subheader("Time-of-Day Accident Trends")
    figure, summary = plot_accidents_by_time()
    st.plotly_chart(figure, use_container_width=True)
    peak_row = summary.loc[summary["number_of_accidents"].idxmax()]
    st.write(
        f"The peak road accident window is {peak_row['time']} with {int(peak_row['number_of_accidents']):,} recorded accidents."
    )

with tabs[4]:
    st.subheader("Monthly Accident Trends")
    figure, summary = plot_accidents_by_month()
    st.plotly_chart(figure, use_container_width=True)
    peak_row = summary.loc[summary["number_of_accidents"].idxmax()]
    st.write(
        f"The busiest accident month in the supporting dataset is {peak_row['month']} with {int(peak_row['number_of_accidents']):,} incidents."
    )
