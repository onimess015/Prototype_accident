"""
Professional Streamlit UI for SafeRoute AI Prediction Results.

This module provides a reusable, feature-rich interface for displaying
machine learning accident risk predictions in a modern dashboard format.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import pandas as pd


class RiskPredictionUI:
    """Professional UI component for accident risk prediction results."""

    # Risk classification boundaries
    RISK_THRESHOLDS = {
        "low": (0.0, 0.39),
        "medium": (0.40, 0.69),
        "high": (0.70, 1.00),
    }

    # Color scheme for risk levels
    COLORS = {
        "low": "#10B981",  # Green
        "medium": "#F59E0B",  # Amber/Orange
        "high": "#EF4444",  # Red
    }

    # Icons for visual enhancement
    ICONS = {
        "low": "✅",
        "medium": "⚠️",
        "high": "🚨",
    }

    # Risk factors based on input features
    RISK_FACTORS = {
        "driver_age": {
            "high_risk": [16, 25, 65, 120],
            "explanation": "Younger (<25) and older (>65) drivers have higher accident rates",
        },
        "alcohol_involvement": {
            "explanation": "Alcohol significantly impairs reaction time and judgment"
        },
        "fatigue": {
            "explanation": "Driver fatigue reduces alertness and reaction speed"
        },
        "mobile_distraction": {
            "explanation": "Mobile phone use increases accident risk by up to 4x"
        },
        "vehicle_speed": {
            "threshold": 80,
            "explanation": "High speeds reduce control and increase severity",
        },
        "rain_fog": {
            "explanation": "Poor visibility conditions increase accident probability"
        },
        "night_driving": {
            "explanation": "Night driving reduces visibility and depth perception"
        },
        "high_traffic": {
            "explanation": "Dense traffic increases collision probability"
        },
    }

    @staticmethod
    def get_risk_category(probability: float) -> str:
        """
        Classify risk probability into category.

        Args:
            probability: Risk probability (0.0 to 1.0)

        Returns:
            Risk category string: 'low', 'medium', or 'high'
        """
        if probability < 0.40:
            return "low"
        elif probability < 0.70:
            return "medium"
        else:
            return "high"

    @staticmethod
    def create_risk_gauge_chart(probability: float) -> go.Figure:
        """
        Create an advanced Plotly gauge chart for risk visualization.

        Args:
            probability: Risk probability (0.0 to 1.0)

        Returns:
            Plotly Figure object
        """
        risk_category = RiskPredictionUI.get_risk_category(probability)

        # Determine gauge color based on risk level
        if risk_category == "low":
            gauge_color = "lightgreen"
        elif risk_category == "medium":
            gauge_color = "gold"
        else:
            gauge_color = "lightcoral"

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Accident Risk Score"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 39], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [40, 69], "color": "rgba(245, 158, 11, 0.2)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 75,
                    },
                },
                number={"suffix": "%", "font": {"size": 20}},
            )
        )

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=12),
        )

        return fig

    @staticmethod
    def get_risk_explanation(
        probability: float,
        driver_age: Optional[int] = None,
        vehicle_speed: Optional[int] = None,
        weather: Optional[str] = None,
        time_of_day: Optional[str] = None,
    ) -> tuple[list, list]:
        """
        Generate risk factors explanation and safety recommendations.

        Args:
            probability: Risk probability
            driver_age: Age of driver
            vehicle_speed: Vehicle speed
            weather: Weather condition
            time_of_day: Time of day

        Returns:
            Tuple of (risk_factors_list, recommendations_list)
        """
        risk_factors = []
        recommendations = []

        # Analyze driver age
        if driver_age:
            if driver_age < 25:
                risk_factors.append("Young driver (<25 years) - Higher accident risk")
                recommendations.append("Take defensive driving courses")
            elif driver_age > 65:
                risk_factors.append(
                    "Senior driver (>65 years) - Age-related risk factors"
                )
                recommendations.append(
                    "Increase following distance and avoid night driving"
                )

        # Analyze speed
        if vehicle_speed and vehicle_speed > 80:
            risk_factors.append(
                f"High speed ({vehicle_speed} km/h) - Reduced vehicle control"
            )
            recommendations.append("Reduce speed and maintain safe following distance")

        # Analyze weather
        if weather:
            if weather.lower() in ["rain", "fog", "snow", "thunderstorm"]:
                risk_factors.append(f"Poor weather ({weather}) - Reduced visibility")
                recommendations.append("Reduce speed and use headlights")

        # Analyze time
        if time_of_day:
            if time_of_day.lower() in ["night", "evening", "dawn"]:
                risk_factors.append(
                    f"Low light conditions ({time_of_day}) - Poor visibility"
                )
                recommendations.append(
                    "Increase alertness and use appropriate lighting"
                )

        # Add generic recommendations
        if probability >= 0.70:
            recommendations.append("Avoid non-essential travel under these conditions")

        if not risk_factors:
            risk_factors.append("Baseline accident risk detected")

        if not recommendations:
            recommendations.append("Maintain safe driving practices")
            recommendations.append("Keep attention on road conditions")

        return risk_factors, recommendations

    @staticmethod
    def render_prediction_result(
        probability: float,
        user_inputs: Optional[Dict[str, Any]] = None,
        show_gauge: bool = True,
    ) -> None:
        """
        Render the complete accident risk prediction result interface.

        Args:
            probability: Predicted risk probability (0.0 to 1.0)
            user_inputs: Dictionary of user input features
            show_gauge: Whether to display the gauge chart
        """
        # Handle non-moving scenario before probability rendering.
        vehicle_speed = user_inputs.get("vehicle_speed") if user_inputs else None
        if vehicle_speed is not None:
            parsed_speed = pd.to_numeric(vehicle_speed, errors="coerce")
            if not pd.isna(parsed_speed) and float(parsed_speed) == 0.0:
                st.warning(
                    "Prediction not possible when vehicle speed is 0 km/h. Please use a speed between 1 and 150 km/h."
                )
                return

        # Validate probability
        if probability is None or not (0 <= probability <= 1):
            st.error("Invalid prediction probability")
            return

        # Add divider
        st.divider()

        # Title section
        st.markdown("## 🚗 Accident Risk Prediction Result")

        # Get risk category and metrics
        risk_category = RiskPredictionUI.get_risk_category(probability)
        risk_percentage = probability * 100
        icon = RiskPredictionUI.ICONS[risk_category]
        color = RiskPredictionUI.COLORS[risk_category]

        # Extract user inputs safely
        driver_age = user_inputs.get("driver_age") if user_inputs else None
        vehicle_speed = user_inputs.get("vehicle_speed") if user_inputs else None
        weather = user_inputs.get("weather_conditions") if user_inputs else None
        time_of_day = user_inputs.get("time_of_day") if user_inputs else None

        # Get risk factors and recommendations
        risk_factors, recommendations = RiskPredictionUI.get_risk_explanation(
            probability,
            driver_age=driver_age,
            vehicle_speed=vehicle_speed,
            weather=weather,
            time_of_day=time_of_day,
        )

        # ==================================================================
        # MAIN LAYOUT: Two Columns
        # ==================================================================
        col1, col2 = st.columns([1, 1.2], gap="large")

        # ==================================================================
        # LEFT COLUMN: Risk Score and Visual Indicators
        # ==================================================================
        with col1:
            st.markdown("### 📊 Risk Assessment")

            # Risk score metric card
            st.metric(
                label="Accident Risk Score",
                value=f"{risk_percentage:.2f}%",
                delta=None,
                help="Probability of accident occurrence (0-100%)",
            )

            # Visual progress bar
            st.markdown("**Risk Level Progression:**")
            st.progress(probability, text=f"{risk_percentage:.1f}%")

            # Risk category badge
            if risk_category == "low":
                st.success(f"{icon} **Low Risk** – Safe driving conditions detected")
            elif risk_category == "medium":
                st.warning(f"{icon} **Medium Risk** – Caution advised")
            else:
                st.error(f"{icon} **High Risk** – Dangerous conditions detected")

            # Gauge chart (optional advanced visualization)
            if show_gauge:
                st.markdown("---")
                st.markdown("**Risk Gauge Visualization:**")
                gauge_chart = RiskPredictionUI.create_risk_gauge_chart(probability)
                st.plotly_chart(
                    gauge_chart, width="stretch", config={"responsive": True}
                )

        # ==================================================================
        # RIGHT COLUMN: Explanation and Recommendations
        # ==================================================================
        with col2:
            # Risk factors section
            st.markdown("### ⚠️ Contributing Risk Factors")

            if risk_factors:
                for i, factor in enumerate(risk_factors, 1):
                    st.markdown(f"• {factor}")
            else:
                st.info("No significant risk factors detected")

            st.markdown("---")

            # Safety recommendations section
            st.markdown("### ✅ Recommended Safety Actions")

            if recommendations:
                for i, recommendation in enumerate(recommendations, 1):
                    st.markdown(f"• **{recommendation}**")
            else:
                st.info("No additional safety measures required")

            parsed_speed = pd.to_numeric(vehicle_speed, errors="coerce")
            if not pd.isna(parsed_speed) and 1 <= float(parsed_speed) <= 30:
                st.info("Low-speed model review active for 1-30 km/h scenarios.")

        # ==================================================================
        # ADDITIONAL INSIGHTS SECTION (Below two columns)
        # ==================================================================
        st.markdown("---")
        st.markdown("### 📈 Detailed Analysis")

        # Create metrics table
        if user_inputs:
            metrics_data = {"Parameter": [], "Value": [], "Risk Impact": []}

            # Driver information
            if "driver_age" in user_inputs:
                metrics_data["Parameter"].append("Driver Age")
                metrics_data["Value"].append(f"{user_inputs['driver_age']} years")
                age = user_inputs["driver_age"]
                if age < 25 or age > 65:
                    metrics_data["Risk Impact"].append("⬆️ Higher")
                else:
                    metrics_data["Risk Impact"].append("⬇️ Standard")

            if "driver_gender" in user_inputs:
                metrics_data["Parameter"].append("Driver Gender")
                metrics_data["Value"].append(user_inputs["driver_gender"])
                metrics_data["Risk Impact"].append("⬇️ Neutral")

            # Vehicle information
            if "vehicle_speed" in user_inputs:
                metrics_data["Parameter"].append("Vehicle Speed")
                metrics_data["Value"].append(f"{user_inputs['vehicle_speed']} km/h")
                speed = user_inputs["vehicle_speed"]
                if speed > 80:
                    metrics_data["Risk Impact"].append("⬆️ Higher")
                else:
                    metrics_data["Risk Impact"].append("⬇️ Standard")

            if "vehicle_type" in user_inputs:
                metrics_data["Parameter"].append("Vehicle Type")
                metrics_data["Value"].append(user_inputs["vehicle_type"])
                metrics_data["Risk Impact"].append("⬇️ Neutral")

            # Road information
            if "road_type" in user_inputs:
                metrics_data["Parameter"].append("Road Type")
                metrics_data["Value"].append(user_inputs["road_type"])
                metrics_data["Risk Impact"].append("⬇️ Varies")

            if "weather_conditions" in user_inputs:
                metrics_data["Parameter"].append("Weather")
                metrics_data["Value"].append(user_inputs["weather_conditions"])
                weather = user_inputs["weather_conditions"].lower()
                if weather in ["rain", "fog", "snow", "thunderstorm"]:
                    metrics_data["Risk Impact"].append("⬆️ Higher")
                else:
                    metrics_data["Risk Impact"].append("⬇️ Clear")

            if "time_of_day" in user_inputs:
                metrics_data["Parameter"].append("Time of Day")
                metrics_data["Value"].append(user_inputs["time_of_day"])
                time = user_inputs["time_of_day"].lower()
                if time in ["night", "evening", "dawn"]:
                    metrics_data["Risk Impact"].append("⬆️ Higher")
                else:
                    metrics_data["Risk Impact"].append("⬇️ Standard")

            # Display metrics table
            if metrics_data["Parameter"]:
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, width="stretch", hide_index=True)

        # ==================================================================
        # FINAL SUMMARY
        # ==================================================================
        st.markdown("---")
        st.markdown("### 📋 Risk Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.markdown(f"**Risk Category**\n{icon} {risk_category.upper()}")

        with summary_col2:
            if probability < 0.40:
                advice = "Safe to drive"
                emoji = "✅"
            elif probability < 0.70:
                advice = "Drive cautiously"
                emoji = "⚠️"
            else:
                advice = "Avoid driving"
                emoji = "🚨"
            st.markdown(f"**Recommendation**\n{emoji} {advice}")

        with summary_col3:
            st.markdown(f"**Confidence**\nHigh confidence prediction")

        st.markdown("---")


# ==============================================================================
# EXAMPLE USAGE FUNCTION
# ==============================================================================


def example_usage():
    """
    Example of how to use the RiskPredictionUI in a Streamlit app.
    """
    st.set_page_config(
        page_title="SafeRoute AI",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 SafeRoute AI - Road Accident Risk Prediction")

    # Example prediction
    example_probability = 0.72

    example_inputs = {
        "driver_age": 28,
        "driver_gender": "Male",
        "alcohol_involvement": True,
        "vehicle_speed": 95,
        "vehicle_type": "SUV",
        "road_type": "Highway",
        "weather_conditions": "Rain",
        "time_of_day": "Night",
    }

    # Render the prediction result
    RiskPredictionUI.render_prediction_result(
        probability=example_probability,
        user_inputs=example_inputs,
        show_gauge=True,
    )


if __name__ == "__main__":
    example_usage()
