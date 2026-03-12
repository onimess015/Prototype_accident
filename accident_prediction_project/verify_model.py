#!/usr/bin/env python
"""Quick verification of the trained model accuracy."""
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now we can import
from utils import MODELS_DIR
from data_loader import load_main_dataset
from preprocessing import clean_dataframe, fit_preprocessor, transform_features
from feature_engineering import select_features_and_target
from sklearn.model_selection import train_test_split


def create_interaction_features(X):
    """Create interaction features."""
    X_enhanced = X.copy()

    if "driver_age" in X.columns and "vehicle_speed" in X.columns:
        X_enhanced["age_speed_interaction"] = X["driver_age"] * X["vehicle_speed"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        X_enhanced["numeric_mean"] = X[numeric_cols].mean(axis=1)
        X_enhanced["numeric_std"] = X[numeric_cols].std(axis=1).fillna(0)

    return X_enhanced


# Load saved artifacts
model_path = MODELS_DIR / "accident_model.pkl"
preprocessor_path = MODELS_DIR / "preprocessor.pkl"

print("=" * 70)
print("MODEL VERIFICATION")
print("=" * 70)

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print(f"\n[OK] Models loaded")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Model name: {getattr(model, 'model_name_', 'unknown')}")

    # Load and prepare test data
    print(f"\n[OK] Loading test data...")
    raw_df = load_main_dataset()
    cleaned_df = clean_dataframe(raw_df)
    X, y = select_features_and_target(cleaned_df)

    # Use same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Add interaction features
    X_test_enhanced = create_interaction_features(X_test)
    X_test_transformed = transform_features(preprocessor, X_test_enhanced)

    # Predict
    y_pred = model.predict(X_test_transformed)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n[OK] Predictions made on {len(y_test)} test samples")
    print(f"\n" + "=" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")

    if accuracy >= 0.75:
        print(f"\n[SUCCESS] TARGET ACHIEVED! Accuracy {accuracy*100:.2f}% >= 75%")
    else:
        print(f"\n[WARNING] Target not met: {accuracy*100:.2f}% (need >= 75%)")

    print("\n" + "=" * 70)

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback

    traceback.print_exc()
