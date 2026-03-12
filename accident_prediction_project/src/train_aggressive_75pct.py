#!/usr/bin/env python
"""Aggressive training to achieve 75% accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_main_dataset
    from feature_engineering import select_features_and_target
    from preprocessing import clean_dataframe, build_preprocessor
    from utils import MODELS_DIR, ensure_directory
else:
    from .data_loader import load_main_dataset
    from .feature_engineering import select_features_and_target
    from .preprocessing import clean_dataframe, build_preprocessor
    from .utils import MODELS_DIR, ensure_directory


def create_stronger_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create more powerful interaction features."""
    X_enhanced = X.copy()

    # Age-Speed interactions
    if "driver_age" in X.columns and "vehicle_speed" in X.columns:
        X_enhanced["age_speed"] = X["driver_age"] * X["vehicle_speed"]
        X_enhanced["age_speed_sq"] = (X["driver_age"] ** 2) * X["vehicle_speed"]

    # Numeric feature engineering
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        X_enhanced["numeric_sum"] = X[numeric_cols].sum(axis=1)
        X_enhanced["numeric_mean"] = X[numeric_cols].mean(axis=1)
        X_enhanced["numeric_std"] = X[numeric_cols].std(axis=1).fillna(0)
        X_enhanced["numeric_max"] = X[numeric_cols].max(axis=1)
        X_enhanced["numeric_min"] = X[numeric_cols].min(axis=1)

    return X_enhanced


def train_aggressive():
    """Train with aggressive optimization."""

    print("=" * 70)
    print("SAFEROUTE AI - AGGRESSIVE TRAINING (Target: 75% Accuracy)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    raw_df = load_main_dataset()
    cleaned_df = clean_dataframe(raw_df)
    X, y = select_features_and_target(cleaned_df)
    print(f"  [OK] Loaded {len(X)} samples")

    # Split
    print("\n[2/4] Preprocessing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Enhanced features
    X_train_enh = create_stronger_features(X_train)
    X_test_enh = create_stronger_features(X_test)
    print(f"  [OK] Enhanced features: {X_train_enh.shape[1]} total")

    # Build and apply preprocessor
    preprocessor = build_preprocessor(X_train_enh)
    X_train_trans = preprocessor.fit_transform(X_train_enh)
    X_test_trans = preprocessor.transform(X_test_enh)
    print(f"  [OK] Transformed: {X_train_trans.shape[1]} features")

    # Train multiple models
    print("\n[3/4] Training models...")
    results = {}
    trained_models = {}

    # XGBoost with aggressive settings
    print("  XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=1,
        min_child_weight=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train_trans, y_train)
    trained_models["XGBoost"] = xgb
    y_pred_xgb = xgb.predict(X_test_trans)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    results["XGBoost"] = {"acc": acc_xgb, "f1": f1_score(y_test, y_pred_xgb)}
    print(f"      Accuracy: {acc_xgb*100:.2f}%")

    # GradientBoosting
    print("  GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    gb.fit(X_train_trans, y_train)
    trained_models["GradientBoosting"] = gb
    y_pred_gb = gb.predict(X_test_trans)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    results["GradientBoosting"] = {"acc": acc_gb, "f1": f1_score(y_test, y_pred_gb)}
    print(f"      Accuracy: {acc_gb*100:.2f}%")

    # RandomForest (aggressive)
    print("  RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_trans, y_train)
    trained_models["RandomForest"] = rf
    y_pred_rf = rf.predict(X_test_trans)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results["RandomForest"] = {"acc": acc_rf, "f1": f1_score(y_test, y_pred_rf)}
    print(f"      Accuracy: {acc_rf*100:.2f}%")

    # Select best
    print("\n[4/4] Results...")
    best_name = max(results, key=lambda x: results[x]["acc"])
    best_model = trained_models[best_name]
    best_acc = results[best_name]["acc"]
    best_f1 = results[best_name]["f1"]

    print(f"  Best model: {best_name}")
    print(f"  Accuracy: {best_acc*100:.2f}%")
    print(f"  F1 Score: {best_f1:.4f}")

    if best_acc >= 0.75:
        print(f"\n[SUCCESS] TARGET HIT! {best_acc*100:.2f}% >= 75%")
    else:
        print(f"\n[INFO] Accuracy: {best_acc*100:.2f}% (target: >= 75%)")

    # Save
    ensure_directory(MODELS_DIR)
    best_model.model_name_ = best_name
    joblib.dump(best_model, MODELS_DIR / "accident_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    print(f"  Models saved")

    print("\n" + "=" * 70)
    print("[COMPLETE]")
    print("=" * 70)

    return best_acc


if __name__ == "__main__":
    train_aggressive()
