#!/usr/bin/env python
"""Test suite for SafeRoute AI project."""

from pathlib import Path
import sys

# Test 1: Training Pipeline
print("=" * 60)
print("TEST 1: Training Pipeline")
print("=" * 60)

try:
    from src.train_model import train_and_save_model

    model, summary = train_and_save_model()
    print("✓ Training pipeline executed successfully")
    print(f"✓ Best model selected: {getattr(model, 'model_name_', 'unknown')}")
    print(f"✓ Best balanced accuracy: {float(summary.loc[0, 'balanced_accuracy']):.4f}")

    # Verify artifacts exist
    models_dir = Path("models")
    assert (models_dir / "accident_model.pkl").exists(), "Model artifact not saved"
    assert (models_dir / "preprocessor.pkl").exists(), "Preprocessor artifact not saved"
    assert (models_dir / "model_metrics.json").exists(), "Metrics report not saved"
    print("✓ Model artifacts saved and verified")
except Exception as e:
    print(f"✗ Training test failed: {e}")
    sys.exit(1)

# Test 2: Prediction Module
print("\n" + "=" * 60)
print("TEST 2: Prediction Module")
print("=" * 60)

try:
    from src.predict import predict_risk

    test_cases = [
        {
            "name": "Low-risk scenario",
            "input": {
                "driver_age": 30,
                "driver_gender": "Female",
                "alcohol_involvement": "No",
                "vehicle_type": "Car",
                "vehicle_speed": 40,
                "road_type": "Urban Road",
                "road_condition": "Dry",
                "lighting_conditions": "Daylight",
                "weather_conditions": "Clear",
                "traffic_control_presence": "Signals",
                "time_of_day": "14:00",
                "day_of_week": "Tuesday",
            },
        },
        {
            "name": "High-risk scenario",
            "input": {
                "driver_age": 60,
                "driver_gender": "Male",
                "alcohol_involvement": "Yes",
                "vehicle_type": "Truck",
                "vehicle_speed": 100,
                "road_type": "National Highway",
                "road_condition": "Wet",
                "lighting_conditions": "Dark",
                "weather_conditions": "Rainy",
                "traffic_control_presence": "None",
                "time_of_day": "22:30",
                "day_of_week": "Saturday",
            },
        },
        {
            "name": "Medium-risk scenario",
            "input": {
                "driver_age": 45,
                "driver_gender": "Male",
                "alcohol_involvement": "No",
                "vehicle_type": "Two-Wheeler",
                "vehicle_speed": 60,
                "road_type": "State Highway",
                "road_condition": "Dry",
                "lighting_conditions": "Dusk",
                "weather_conditions": "Foggy",
                "traffic_control_presence": "Signs",
                "time_of_day": "18:15",
                "day_of_week": "Friday",
            },
        },
    ]

    for test in test_cases:
        result = predict_risk(test["input"])
        print(f"\n{test['name']}:")
        print(f"  Risk Score: {result['risk_score']:.4f}")
        print(f"  Risk Percentage: {result['risk_percentage']:.2f}%")
        print(f"  Risk Label: {result['risk_label']}")

        # Validate result structure
        assert "risk_score" in result, "Missing risk_score in result"
        assert "risk_percentage" in result, "Missing risk_percentage in result"
        assert "risk_label" in result, "Missing risk_label in result"
        assert 0 <= result["risk_score"] <= 1, "Risk score out of valid range [0, 1]"
        assert result["risk_label"] in [
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ], "Invalid risk label"
        print(f"  ✓ Prediction valid")

    print("\n✓ Prediction module test passed")
except Exception as e:
    print(f"✗ Prediction test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Analytics & Data Loading
print("\n" + "=" * 60)
print("TEST 3: Analytics & Data Loading")
print("=" * 60)

try:
    from src.analytics import (
        plot_accidents_by_cause,
        plot_accidents_by_road_type,
        plot_accidents_by_time,
        plot_accidents_by_month,
    )

    analytics_functions = [
        ("plot_accidents_by_cause", plot_accidents_by_cause),
        ("plot_accidents_by_road_type", plot_accidents_by_road_type),
        ("plot_accidents_by_time", plot_accidents_by_time),
        ("plot_accidents_by_month", plot_accidents_by_month),
    ]

    for func_name, func in analytics_functions:
        figure, summary = func()
        assert figure is not None, f"{func_name} returned None figure"
        assert summary is not None, f"{func_name} returned None summary"
        print(f"✓ {func_name} executed successfully ({len(summary)} rows)")

    print("\n✓ Analytics module test passed")
except Exception as e:
    print(f"✗ Analytics test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Feature Engineering
print("\n" + "=" * 60)
print("TEST 4: Feature Engineering")
print("=" * 60)

try:
    from src.feature_engineering import get_feature_schema, select_features_and_target
    from src.data_loader import load_main_dataset
    from src.preprocessing import clean_dataframe

    schema = get_feature_schema()
    print(f"✓ Feature schema obtained: {len(schema)} features")
    assert len(schema) > 0, "Feature schema is empty"

    # Verify features are the expected ones
    expected_features = [
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

    for feature in expected_features:
        assert feature in schema, f"Expected feature '{feature}' not in schema"

    print(f"✓ All expected features verified: {expected_features}")

    # Verify feature engineering works on real data
    df = clean_dataframe(load_main_dataset())
    X, y = select_features_and_target(df)
    assert len(X) > 0, "Feature matrix is empty"
    assert len(y) > 0, "Target vector is empty"
    assert len(X) == len(y), "Feature matrix and target have different lengths"
    print(
        f"✓ Feature extraction from dataset: {len(X)} samples with {len(X.columns)} features"
    )

    print("\n✓ Feature engineering test passed")
except Exception as e:
    print(f"✗ Feature engineering test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "=" * 60)
print("TESTING SUMMARY")
print("=" * 60)
print("✓ All tests passed successfully!")
print("\nProject is ready for deployment.")
print("\nTo run the Streamlit app:")
print("  streamlit run app/streamlit_app.py")
print("=" * 60)
