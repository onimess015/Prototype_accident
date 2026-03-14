from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_main_dataset  # noqa: E402
from src.predict import predict_risk  # noqa: E402
from src.train_model import train_and_save_model  # noqa: E402
from src.utils import MODELS_DIR  # noqa: E402


st.set_page_config(page_title="SafeRoute AI", page_icon="AI", layout="wide")


WEATHER_OPTIONS = ["Clear", "Rain", "Fog", "Snow"]
ROAD_OPTIONS = ["Highway", "City", "Rural"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High"]
TIME_OPTIONS = ["Day", "Night"]
FATIGUE_OPTIONS = ["Low", "Medium", "High"]
LIGHTING_OPTIONS = ["Good", "Moderate", "Poor"]


def derive_visibility_level(weather: str, road_lighting: str) -> str:
    weather_text = str(weather).strip().lower()
    lighting_text = str(road_lighting).strip().lower()

    if weather_text in {"rain", "fog", "snow"}:
        return "Low" if lighting_text in {"poor", "dark", "dusk", "dawn"} else "Medium"
    if lighting_text in {"poor", "dark", "dusk", "dawn"}:
        return "Medium"
    return "High"


def render_explanation_cards(result: dict[str, object]) -> None:
    st.markdown(
        """
                <style>
                .scenario-grid {
                        display: grid;
                        grid-template-columns: repeat(2, minmax(0, 1fr));
                        gap: 0.8rem;
                        margin-top: 0.6rem;
                }
                .scenario-card {
                        border: 1px solid #e5e7eb;
                        border-radius: 0.8rem;
                        padding: 0.9rem;
                        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                }
                .scenario-card h4 {
                        margin: 0 0 0.35rem 0;
                        font-size: 0.95rem;
                        color: #1f2937;
                }
                .scenario-card p {
                        margin: 0;
                        color: #374151;
                        font-size: 0.92rem;
                        line-height: 1.35;
                }
                @media (max-width: 880px) {
                        .scenario-grid {
                                grid-template-columns: 1fr;
                        }
                }
                </style>
                """,
        unsafe_allow_html=True,
    )

    reason = str(result.get("accident_reason", "Reason unavailable"))
    cause = str(result.get("accident_cause", "Cause unavailable"))
    why = str(result.get("accident_why", "Why unavailable"))
    how = str(result.get("accident_how", "How unavailable"))

    cards_html = f"""
        <div class="scenario-grid">
            <div class="scenario-card">
                <h4>Reason</h4>
                <p>{reason}</p>
            </div>
            <div class="scenario-card">
                <h4>Cause</h4>
                <p>{cause}</p>
            </div>
            <div class="scenario-card">
                <h4>Why</h4>
                <p>{why}</p>
            </div>
            <div class="scenario-card">
                <h4>How</h4>
                <p>{how}</p>
            </div>
        </div>
        """

    st.markdown(cards_html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return load_main_dataset()


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    path = MODELS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["feature", "importance"])


@st.cache_data(show_spinner=False)
def load_model_summary() -> pd.DataFrame:
    path = MODELS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def ensure_model_ready() -> str:
    model_path = MODELS_DIR / "model.pkl"
    if model_path.exists():
        return "loaded"

    train_and_save_model()
    return "trained"


def risk_color(label: str) -> str:
    if label == "Low Risk":
        return "#15803d"
    if label == "Medium Risk":
        return "#b45309"
    return "#b91c1c"
APP_NOTE = (
    "SafeRoute AI is an AI-powered road safety dashboard that predicts accident risk "
    "using driving and environmental conditions such as speed, weather, road type, "
    "traffic density, lighting, and driver fatigue. The system not only estimates risk "
    "but also explains why risk is elevated, highlights major contributing factors, "
    "and helps users explore different conditions through interactive scenario simulation. "
    "The goal is to make accident-risk analysis easy to understand, visually engaging, "
    "and useful for real-world decision support."
)

def risk_gauge(probability: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Accident Risk Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f2937"},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
            },
        )
    ).update_layout(height=280, margin={"l": 10, "r": 10, "t": 45, "b": 10})


def probability_bar(probability: float) -> go.Figure:
    safe_probability = max(0.0, min(1.0, probability))
    return px.bar(
        pd.DataFrame(
            {
                "class": ["Low Outcome", "Severe Outcome"],
                "probability": [1 - safe_probability, safe_probability],
            }
        ),
        x="class",
        y="probability",
        color="class",
        color_discrete_map={
            "Low Outcome": "#22c55e",
            "Severe Outcome": "#ef4444",
        },
        title="Prediction Probability Breakdown",
    ).update_layout(showlegend=False, yaxis_title="Probability")


def validate_batch_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "vehicle_speed",
        "weather_condition",
        "road_type",
        "traffic_density",
        "time_of_day",
        "driver_fatigue",
        "road_lighting",
    ]
    normalized = df.copy()
    normalized.columns = [
        str(c).strip().lower().replace(" ", "_") for c in normalized.columns
    ]

    missing = [column for column in required if column not in normalized.columns]
    if missing:
        return normalized, missing

    normalized["vehicle_speed"] = pd.to_numeric(
        normalized["vehicle_speed"], errors="coerce"
    )
    normalized = normalized.dropna(subset=["vehicle_speed"]).copy()
    normalized["vehicle_speed"] = normalized["vehicle_speed"].clip(0, 150)

    if "visibility_level" not in normalized.columns:
        normalized["visibility_level"] = [
            derive_visibility_level(weather, lighting)
            for weather, lighting in zip(
                normalized["weather_condition"], normalized["road_lighting"]
            )
        ]

    return normalized, []


def show_intro_section() -> None:
    st.title("SafeRoute AI")
    st.caption(
        "Portfolio-grade accident risk prediction with explainable ML and Streamlit optimization"
    )

    st.markdown(
        """
        This dashboard predicts accident risk probability using a trained machine learning pipeline.
        It includes input validation, explainability, batch prediction, and visual analytics.
        """
    )


def show_prediction_section() -> None:
    st.subheader("Risk Prediction Dashboard")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        speed = st.slider("Vehicle Speed (km/h)", min_value=0, max_value=150, value=60)
        weather = st.selectbox("Weather", WEATHER_OPTIONS)
    with col_b:
        road_type = st.selectbox("Road Type", ROAD_OPTIONS)
        traffic_density = st.selectbox("Traffic Density", TRAFFIC_OPTIONS)
    with col_c:
        time_of_day = st.selectbox("Time of Day", TIME_OPTIONS)
        driver_fatigue = st.selectbox("Driver Fatigue", FATIGUE_OPTIONS)
    with col_d:
        road_lighting = st.selectbox("Road Lighting", LIGHTING_OPTIONS)

    if speed < 0 or speed > 150:
        st.error("Speed must be between 0 and 150 km/h.")
        return

    payload = {
        "vehicle_speed": speed,
        "weather_condition": weather,
        "road_type": road_type,
        "traffic_density": traffic_density,
        "time_of_day": time_of_day,
        "driver_fatigue": driver_fatigue,
        "road_lighting": road_lighting,
        "visibility_level": derive_visibility_level(weather, road_lighting),
    }

    if st.button("Predict Risk", type="primary"):
        if speed == 0:
            st.warning(
                "Prediction not possible when speed is 0 km/h. Please set speed between 1 and 150 km/h."
            )
            return

        try:
            result = predict_risk(payload)

            if not bool(result.get("prediction_possible", True)):
                st.warning(
                    str(
                        result.get(
                            "prediction_message",
                            "Prediction not possible for this scenario.",
                        )
                    )
                )
                return

            probability = float(result["risk_score"])
            label = str(result["risk_label"])
            color = risk_color(label)

            st.progress(probability, text=f"Risk score: {result['risk_percentage']}%")
            st.markdown(
                f"<div style='padding:0.8rem;border-radius:0.6rem;background:{color};color:white;font-weight:700;'>Risk Level: {label}</div>",
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)
            with left:
                st.plotly_chart(risk_gauge(probability), width="stretch")
            with right:
                st.plotly_chart(probability_bar(probability), width="stretch")

            factors = result.get("top_factors", [])
            if factors:
                st.markdown("Top Risk Factors")
                for idx, factor in enumerate(factors, start=1):
                    st.write(f"{idx}. {factor}")

            st.markdown("### Scenario Explanation")
            st.caption(f"Scenario ID: {result.get('scenario_signature', 'n/a')}")
            render_explanation_cards(result)

            if bool(result.get("low_speed_review_applied", False)):
                st.info("Low-speed model review applied for 1-30 km/h input.")

            history = st.session_state.get("prediction_history", [])
            user_facing_payload = {
                k: v for k, v in payload.items() if k != "visibility_level"
            }
            history.append(
                {
                    **user_facing_payload,
                    "risk_probability": result["risk_score"],
                    "risk_level": result["risk_label"],
                    "scenario_signature": result.get("scenario_signature", "n/a"),
                    "low_speed_review_applied": bool(
                        result.get("low_speed_review_applied", False)
                    ),
                }
            )
            st.session_state["prediction_history"] = history[-100:]

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def show_model_explanation_section() -> None:
    st.subheader("Model Explanation")

    comparison = load_model_summary()
    if not comparison.empty:
        st.markdown("Model Benchmark")
        st.dataframe(comparison, width="stretch")

    importance = load_feature_importance()
    if not importance.empty:
        fig = px.bar(
            importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Importances",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Feature importance will appear after training the model.")


def show_dataset_section() -> None:
    st.subheader("About the Dataset")
    df = load_dataset()
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df.head(20), width="stretch")

    if {"state_name", "city_name"}.issubset({c.lower() for c in df.columns}):
        normalized = df.copy()
        normalized.columns = [str(c).strip().lower() for c in normalized.columns]
        hotspot = (
            normalized.groupby(["state_name", "city_name"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(20)
        )
        st.markdown("Accident Hotspots (Top State-City Pairs)")
        st.dataframe(hotspot, width="stretch")


def show_advanced_section() -> None:
    st.subheader("Advanced Features")

    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        batch_df, missing = validate_batch_frame(uploaded_df)
        if missing:
            st.error(
                f"Missing required columns for batch prediction: {', '.join(missing)}"
            )
        else:
            predictions = []
            for _, row in batch_df.iterrows():
                row_payload = {
                    "vehicle_speed": float(row["vehicle_speed"]),
                    "weather_condition": str(row["weather_condition"]),
                    "road_type": str(row["road_type"]),
                    "traffic_density": str(row["traffic_density"]),
                    "time_of_day": str(row["time_of_day"]),
                    "driver_fatigue": str(row["driver_fatigue"]),
                    "road_lighting": str(row["road_lighting"]),
                    "visibility_level": str(row["visibility_level"]),
                }
                pred = predict_risk(row_payload)
                if not bool(pred.get("prediction_possible", True)):
                    predictions.append(
                        {
                            **row_payload,
                            "risk_probability": None,
                            "risk_level": "Prediction Not Possible",
                            "prediction_message": str(
                                pred.get(
                                    "prediction_message",
                                    "Prediction not possible for this scenario.",
                                )
                            ),
                            "low_speed_review_applied": False,
                        }
                    )
                    continue

                predictions.append(
                    {
                        **row_payload,
                        "risk_probability": pred["risk_score"],
                        "risk_level": pred["risk_label"],
                        "prediction_message": str(pred.get("prediction_message", "")),
                        "low_speed_review_applied": bool(
                            pred.get("low_speed_review_applied", False)
                        ),
                    }
                )

            result_df = pd.DataFrame(predictions)
            if "visibility_level" in result_df.columns:
                result_df = result_df.drop(columns=["visibility_level"])
            st.dataframe(result_df.head(50), width="stretch")

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Batch Prediction Report",
                data=io.BytesIO(csv_bytes),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

    history = st.session_state.get("prediction_history", [])
    if history:
        history_df = pd.DataFrame(history)
        st.markdown("Prediction History")
        st.dataframe(history_df.tail(20), width="stretch")

        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction History",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )


def main() -> None:
    ensure_model_ready()

    show_intro_section()

    tabs = st.tabs(
        [
            "Risk Prediction",
            "Model Explanation",
            "About Dataset",
            "Advanced",
        ]
    )

    with tabs[0]:
        show_prediction_section()
    with tabs[1]:
        show_model_explanation_section()
    with tabs[2]:
        show_dataset_section()
    with tabs[3]:
        show_advanced_section()


if __name__ == "__main__":
    main()
