#!/usr/bin/env python
"""Targeted tests for speed edge-case prediction behavior."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.predict import predict_risk


BASE_INPUT = {
    "driver_age": 35,
    "driver_gender": "Male",
    "alcohol_involvement": "No",
    "vehicle_type": "Car",
    "road_type": "Urban",
    "road_condition": "Dry",
    "lighting_conditions": "Daylight",
    "weather_conditions": "Clear",
    "time_of_day": "Morning",
    "traffic_control_presence": "Yes",
    "day_of_week": "Monday",
}


def _predict_for_speed(speed_kmh: int) -> dict:
    payload = dict(BASE_INPUT)
    payload["vehicle_speed"] = speed_kmh
    return predict_risk(payload)


def test_zero_speed_prediction_not_possible() -> None:
    result = _predict_for_speed(0)
    assert result.get("prediction_possible") is False
    assert result.get("risk_label") == "Prediction Not Possible"
    assert "0 km/h" in str(result.get("prediction_message", ""))


def test_low_speed_review_applied_from_1_to_30() -> None:
    for speed in (1, 15, 30):
        result = _predict_for_speed(speed)
        assert result.get("prediction_possible") is True
        assert result.get("low_speed_review_applied") is True
        assert "1-30" in str(result.get("prediction_message", ""))
        score = float(result.get("risk_score", -1))
        assert 0.0 <= score <= 1.0


def test_above_low_speed_range_no_review() -> None:
    for speed in (31, 40):
        result = _predict_for_speed(speed)
        assert result.get("prediction_possible") is True
        assert result.get("low_speed_review_applied") is False


def main() -> None:
    test_zero_speed_prediction_not_possible()
    test_low_speed_review_applied_from_1_to_30()
    test_above_low_speed_range_no_review()
    print("[OK] Speed edge-case tests passed")


if __name__ == "__main__":
    main()
