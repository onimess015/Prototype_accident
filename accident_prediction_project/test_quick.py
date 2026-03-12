#!/usr/bin/env python
"""Quick test of SafeRoute AI project."""

import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("SAFEROUTE AI - TEST SUITE")
print("=" * 70)

print("\n[TEST 1/5] Module imports...")
try:
    from src.data_loader import load_main_dataset
    from src.preprocessing import clean_dataframe
    from src.feature_engineering import select_features_and_target
    from src.predict import load_artifacts, predict_risk

    print("  [OK] Imports successful")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("\n[TEST 2/5] Data loading...")
try:
    df = load_main_dataset()
    print(f"  [OK] Loaded {df.shape[0]} rows, {df.shape[1]} cols")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("\n[TEST 3/5] Data preprocessing...")
try:
    cleaned_df = clean_dataframe(df)
    X, y = select_features_and_target(cleaned_df)
    print(f"  [OK] Features: {X.shape[1]}, Target: 0={sum(y==0)}, 1={sum(y==1)}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("\n[TEST 4/5] Model loading...")
try:
    model, preprocessor = load_artifacts()
    print(f"  [OK] Model: {type(model).__name__}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("\n[TEST 5/5] Prediction test...")
try:
    test_input = {
        "driver_age": 35,
        "driver_gender": "Male",
        "alcohol_involvement": False,
        "vehicle_speed": 60,
        "vehicle_type": "Car",
        "road_type": "Urban",
        "road_condition": "Dry",
        "lighting_conditions": "Daylight",
        "weather_conditions": "Clear",
        "time_of_day": "Morning",
        "traffic_control_presence": "Yes",
        "day_of_week": "Monday",
    }
    pred = predict_risk(test_input)
    print(f"  [OK] Risk: {pred['risk_percentage']:.2f}% ({pred['risk_label']})")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print("\nTo run Streamlit dashboard:")
print("  cd c:\\Prototype\\(accident_predictor)\\accident_prediction_project")
print("  streamlit run app/streamlit_app.py")
print("\n" + "=" * 70)
