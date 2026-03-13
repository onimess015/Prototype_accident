#!/usr/bin/env python
"""Comprehensive tests for SafeRoute AI project."""

import importlib.util
import json
import sys
from pathlib import Path

# Set up proper import paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

print("=" * 70)
print("SAFEROUTE AI - COMPREHENSIVE TEST SUITE")
print("=" * 70)

# Test 1: Module imports
print("\n[TEST 1/6] Testing module imports...")
try:
    from src.data_loader import load_main_dataset
    from src.preprocessing import clean_dataframe
    from src.feature_engineering import select_features_and_target
    from src.predict import load_artifacts, predict_risk
    from src.preprocessing import build_preprocessor

    print("  [OK] All core modules imported successfully")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Data loading
print("\n[TEST 2/6] Testing data loading...")
try:
    raw_df = load_main_dataset()
    print(f"  [OK] Dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
except Exception as e:
    print(f"  [ERROR] Data loading failed: {e}")
    sys.exit(1)

# Test 3: Data preprocessing
print("\n[TEST 3/6] Testing data preprocessing...")
try:
    cleaned_df = clean_dataframe(raw_df)
    X, y = select_features_and_target(cleaned_df)
    print(f"  [OK] Features engineered: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  [OK] Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
except Exception as e:
    print(f"  [ERROR] Preprocessing failed: {e}")
    sys.exit(1)

# Test 4: Model artifacts
print("\n[TEST 4/6] Testing model artifacts...")
try:
    model, preprocessor = load_artifacts()
    print(f"  [OK] Model loaded: {type(model).__name__}")
    print(f"  [OK] Preprocessor loaded")
except Exception as e:
    print(f"  [ERROR] Model loading failed: {e}")
    sys.exit(1)

# Test 5: Prediction functionality
print("\n[TEST 5/6] Testing prediction functionality...")
try:
    test_inputs = {
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

    prediction = predict_risk(test_inputs)
    print(f"  [OK] Prediction successful")
    print(f"      Risk Score: {prediction['risk_score']:.4f}")
    print(f"      Risk Percentage: {prediction['risk_percentage']:.2f}%")
    print(f"      Risk Label: {prediction['risk_label']}")
except Exception as e:
    print(f"  [ERROR] Prediction failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Streamlit UI component
print("\n[TEST 6/6] Testing Streamlit UI component...")
try:
    # Just check if imports work - can't run full Streamlit test in script
    ui_file = project_root / "app" / "prediction_ui.py"
    module_spec = importlib.util.spec_from_file_location("prediction_ui", ui_file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to load UI module from {ui_file}")
    ui_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(ui_module)
    RiskPredictionUI = ui_module.RiskPredictionUI

    # Test UI methods exist
    assert hasattr(RiskPredictionUI, "get_risk_category")
    assert hasattr(RiskPredictionUI, "create_risk_gauge_chart")
    assert hasattr(RiskPredictionUI, "get_risk_explanation")
    assert hasattr(RiskPredictionUI, "render_prediction_result")

    # Test risk classification
    assert RiskPredictionUI.get_risk_category(0.30) == "low"
    assert RiskPredictionUI.get_risk_category(0.55) == "medium"
    assert RiskPredictionUI.get_risk_category(0.80) == "high"

    print(f"  [OK] UI component loaded and verified")
    print(f"  [OK] Risk classification working correctly")
except Exception as e:
    print(f"  [ERROR] UI component test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED - PROJECT READY FOR DEPLOYMENT")
print("=" * 70)
print("\nProject Statistics:")
metrics_file = project_root / "models" / "model_metrics.json"
if metrics_file.exists():
    metrics_report = json.loads(metrics_file.read_text(encoding="utf-8"))
    best_model_name = metrics_report.get("best_model", "unknown")
    metrics_rows = metrics_report.get("metrics", [])
    best_row = metrics_rows[0] if metrics_rows else {}
else:
    best_model_name = "unknown"
    best_row = {}

print(f"  Model: {best_model_name}")
print(f"  Dataset size: {raw_df.shape[0]} samples")
print(f"  Features: {X.shape[1]} engineered features")
if best_row:
    test_accuracy = float(best_row.get("accuracy", 0.0))
    print(f"  Test accuracy: {test_accuracy * 100:.2f}%")
    if "balanced_accuracy" in best_row:
        balanced_accuracy = float(best_row.get("balanced_accuracy", 0.0))
        print(f"  Test balanced accuracy: {balanced_accuracy * 100:.2f}%")
        assert (
            balanced_accuracy >= 0.50
        ), f"Balanced accuracy too low: {balanced_accuracy:.4f}"
    specificity = float(best_row.get("specificity", 0.0))
    print(f"  Test specificity: {specificity * 100:.2f}%")
    assert specificity >= 0.10, f"Specificity too low: {specificity:.4f}"
    assert not bool(
        best_row.get("degenerate_prediction_flag", False)
    ), "Degenerate prediction behavior detected"
print(f"  Dashboard: Streamlit with professional UI")
print("\nTo launch the dashboard:")
print("  python start_streamlit.py")
print("\n" + "=" * 70)
