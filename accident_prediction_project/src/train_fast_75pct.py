#!/usr/bin/env python
"""Fast training to achieve 75% accuracy target."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Handle both module and direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_main_dataset
    from feature_engineering import select_features_and_target
    from preprocessing import clean_dataframe, fit_preprocessor, transform_features
    from utils import MODELS_DIR, ensure_directory
else:
    from .data_loader import load_main_dataset
    from .feature_engineering import select_features_and_target
    from .preprocessing import clean_dataframe, fit_preprocessor, transform_features
    from .utils import MODELS_DIR, ensure_directory


def create_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    X_enhanced = X.copy()

    if "driver_age" in X.columns and "vehicle_speed" in X.columns:
        X_enhanced["age_speed_interaction"] = X["driver_age"] * X["vehicle_speed"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        X_enhanced["numeric_mean"] = X[numeric_cols].mean(axis=1)
        X_enhanced["numeric_std"] = X[numeric_cols].std(axis=1).fillna(0)

    return X_enhanced


def train_for_75_percent():
    """Train model focused on achieving 75% accuracy."""

    print("=" * 70)
    print("SAFEROUTE AI - FAST TRAINING (Target: 75% Accuracy)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    raw_df = load_main_dataset()
    cleaned_df = clean_dataframe(raw_df)
    X, y = select_features_and_target(cleaned_df)
    print(f"  [OK] Loaded {len(X)} samples, {X.shape[1]} features")
    print(f"  [OK] Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")

    # Split and preprocess
    print("\n[2/4] Preprocessing features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Add interaction features
    X_train_enhanced = create_interaction_features(X_train)
    X_test_enhanced = create_interaction_features(X_test)

    # Fit preprocessor
    preprocessor = fit_preprocessor(X_train_enhanced)
    X_train_transformed = transform_features(preprocessor, X_train_enhanced)
    X_test_transformed = transform_features(preprocessor, X_test_enhanced)
    print(f"  [OK] Transformed to {X_train_transformed.shape[1]} features")

    # Train models with optimized parameters
    print("\n[3/4] Training optimized models...")
    results = {}

    # XGBoost - often best for tabular data
    print("  Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train_transformed, y_train)
    y_pred_xgb = xgb.predict(X_test_transformed)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    results["XGBoost"] = {
        "accuracy": acc_xgb,
        "f1": f1_xgb,
        "precision": precision_score(y_test, y_pred_xgb),
        "recall": recall_score(y_test, y_pred_xgb),
    }
    print(f"    Accuracy: {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")

    # Random Forest with balanced weights
    print("  Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_transformed, y_train)
    y_pred_rf = rf.predict(X_test_transformed)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    results["RandomForest"] = {
        "accuracy": acc_rf,
        "f1": f1_rf,
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
    }
    print(f"    Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

    # Select best model
    print("\n[4/4] Selecting best model...")
    best_model_name = "XGBoost" if acc_xgb >= acc_rf else "RandomForest"
    best_model = xgb if best_model_name == "XGBoost" else rf
    best_accuracy = results[best_model_name]["accuracy"]

    print(f"  [OK] Best model: {best_model_name}")
    print(f"  [OK] Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"  [OK] F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"  [OK] Precision: {results[best_model_name]['precision']:.4f}")
    print(f"  [OK] Recall: {results[best_model_name]['recall']:.4f}")

    if best_accuracy >= 0.75:
        print(f"\n[SUCCESS] TARGET MET! Accuracy {best_accuracy*100:.2f}% >= 75%")
    else:
        print(
            f"\n[WARNING] Target not fully met: {best_accuracy*100:.2f}% (need >= 75%)"
        )

    # Save model
    ensure_directory(MODELS_DIR)
    best_model.model_name_ = best_model_name
    joblib.dump(best_model, MODELS_DIR / "accident_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    print(f"\n  [OK] Model saved")

    print("\n" + "=" * 70)
    print("[SUCCESS] TRAINING COMPLETE")
    print("=" * 70)

    return best_model, preprocessor, best_accuracy


if __name__ == "__main__":
    train_for_75_percent()
