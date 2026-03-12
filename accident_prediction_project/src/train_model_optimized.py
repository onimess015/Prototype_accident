#!/usr/bin/env python
"""Optimized training pipeline for SafeRoute AI with improved accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

# Handle both module and direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_main_dataset
    from evaluate_model import compare_models, evaluate_classifier
    from feature_engineering import select_features_and_target
    from preprocessing import clean_dataframe, fit_preprocessor, transform_features
    from utils import MODELS_DIR, ensure_directory
else:
    from .data_loader import load_main_dataset
    from .evaluate_model import compare_models, evaluate_classifier
    from .feature_engineering import select_features_and_target
    from .preprocessing import clean_dataframe, fit_preprocessor, transform_features
    from .utils import MODELS_DIR, ensure_directory


def create_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features to improve model performance."""
    X_enhanced = X.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if "driver_age" in numeric_cols and "vehicle_speed" in numeric_cols:
        X_enhanced["age_speed_interaction"] = X["driver_age"] * X["vehicle_speed"]

    if len(numeric_cols) >= 2:
        X_enhanced["numeric_mean"] = X[numeric_cols].mean(axis=1)
        X_enhanced["numeric_std"] = X[numeric_cols].std(axis=1)

    return X_enhanced


def train_and_optimize_model() -> tuple[object, object]:
    """Train and optimize the model for maximum accuracy."""

    print("=" * 70)
    print("SAFEROUTE AI - OPTIMIZED TRAINING (Target: >75% Accuracy)")
    print("=" * 70)

    # 1. Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    raw_dataframe = load_main_dataset()
    cleaned_dataframe = clean_dataframe(raw_dataframe)
    X, y = select_features_and_target(cleaned_dataframe)

    print(f"  [OK] Dataset shape: {X.shape}")
    print(f"  [OK] Target distribution: {dict(y.value_counts())}")
    print(
        f"  [OK] Class imbalance ratio: {y.value_counts()[1] / y.value_counts()[0]:.2f}"
    )

    # 2. Split data with stratification
    print("\n[2/5] Splitting data...")
    stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    # Create interaction features
    X_train_enhanced = create_interaction_features(X_train)
    X_test_enhanced = create_interaction_features(X_test)

    print(f"  [OK] Train set: {X_train_enhanced.shape}")
    print(f"  [OK] Test set: {X_test_enhanced.shape}")

    # 3. Preprocess features
    print("\n[3/5] Preprocessing features...")
    preprocessor = fit_preprocessor(X_train_enhanced)
    preprocessor.feature_schema_ = list(X_train_enhanced.columns)
    X_train_transformed = transform_features(preprocessor, X_train_enhanced)
    X_test_transformed = transform_features(preprocessor, X_test_enhanced)

    print(f"  [OK] Transformed features: {X_train_transformed.shape[1]}")

    # 4. Train optimized models with hyperparameter tuning
    print("\n[4/5] Training optimized models with hyperparameter tuning...")

    results: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    # Model 1: Optimized RandomForest (balanced for class imbalance)
    print("\n  Training RandomForest with optimization...")
    rf_params = {
        "n_estimators": [300, 500],
        "max_depth": [10, 15, 20],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "class_weight": ["balanced"],
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring="accuracy", n_jobs=-1)
    rf_grid.fit(X_train_transformed, y_train)

    rf_model = rf_grid.best_estimator_
    print(f"    [OK] Best params: {rf_grid.best_params_}")
    print(f"    [OK] CV accuracy: {rf_grid.best_score_:.4f}")

    trained_models["RandomForest"] = rf_model
    metrics_rf = evaluate_classifier(rf_model, X_test_transformed, y_test)
    results["RandomForest"] = metrics_rf

    # Model 2: GradientBoosting (often better accuracy)
    print("\n  Training GradientBoosting with optimization...")
    gb_params = {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 5, 6],
        "subsample": [0.8, 0.9],
    }

    gb_base = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb_base, gb_params, cv=5, scoring="accuracy", n_jobs=-1)
    gb_grid.fit(X_train_transformed, y_train)

    gb_model = gb_grid.best_estimator_
    print(f"    [OK] Best params: {gb_grid.best_params_}")
    print(f"    [OK] CV accuracy: {gb_grid.best_score_:.4f}")

    trained_models["GradientBoosting"] = gb_model
    metrics_gb = evaluate_classifier(gb_model, X_test_transformed, y_test)
    results["GradientBoosting"] = metrics_gb

    # Model 3: Optimized XGBoost
    print("\n  Training XGBoost with optimization...")
    xgb_params = {
        "n_estimators": [200, 300],
        "max_depth": [5, 6, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 0.9],
        "scale_pos_weight": [y_train.value_counts()[0] / y_train.value_counts()[1]],
    }

    xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)
    xgb_grid = GridSearchCV(xgb_base, xgb_params, cv=5, scoring="accuracy", n_jobs=-1)
    xgb_grid.fit(X_train_transformed, y_train)

    xgb_model = xgb_grid.best_estimator_
    print(f"    [OK] Best params: {xgb_grid.best_params_}")
    print(f"    [OK] CV accuracy: {xgb_grid.best_score_:.4f}")

    trained_models["XGBoost"] = xgb_model
    metrics_xgb = evaluate_classifier(xgb_model, X_test_transformed, y_test)
    results["XGBoost"] = metrics_xgb

    # 5. Select and save best model
    print("\n[5/5] Selecting best model...")

    comparison = compare_models(results)
    best_model_name = comparison.loc[0, "model"]
    best_model = trained_models[best_model_name]
    best_model.model_name_ = best_model_name

    best_accuracy = comparison.loc[0, "accuracy"]
    best_f1 = comparison.loc[0, "f1"]

    print(f"\n  [OK] Best model selected: {best_model_name}")
    print(f"    - Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"    - F1 Score: {best_f1:.4f}")

    # Check if target is met
    if best_accuracy > 0.75:
        print(f"\n  [SUCCESS] TARGET MET: Accuracy {best_accuracy*100:.2f}% > 75%!")
    else:
        print(f"\n  [WARNING] Target not met: {best_accuracy*100:.2f}% (need > 75%)")

    # Save artifacts
    ensure_directory(MODELS_DIR)
    joblib.dump(best_model, MODELS_DIR / "accident_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    print(f"\n  [OK] Model saved to: {MODELS_DIR / 'accident_model.pkl'}")
    print(f"  [OK] Preprocessor saved to: {MODELS_DIR / 'preprocessor.pkl'}")

    print("\n" + "=" * 70)
    print("[OK] TRAINING COMPLETE")
    print("=" * 70)

    return best_model, preprocessor


if __name__ == "__main__":
    train_and_optimize_model()
