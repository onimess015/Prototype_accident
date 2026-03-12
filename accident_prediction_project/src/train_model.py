from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_preprocessing import REQUIRED_MODEL_FEATURES, load_and_prepare_data
from .utils import MODELS_DIR, ensure_directory


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn compatibility fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", _make_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, object]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def _extract_feature_importance(best_model: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out().tolist()

    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        importance_values = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        importance_values = np.zeros(len(feature_names), dtype=float)

    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)

    return importance_frame.reset_index(drop=True)


def train_and_save_model() -> tuple[Pipeline, pd.DataFrame]:
    prepared = load_and_prepare_data(test_size=0.2, random_state=42)
    X_train, X_test = prepared.X_train, prepared.X_test
    y_train, y_test = prepared.y_train, prepared.y_test

    numeric_features = ["vehicle_speed"]
    categorical_features = [
        feature
        for feature in REQUIRED_MODEL_FEATURES
        if feature not in numeric_features
    ]

    candidate_models = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
            random_state=42,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    evaluation_rows: list[dict[str, object]] = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    _build_preprocessor(
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    ),
                ),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = _evaluate_model(pipeline, X_test, y_test)
        metrics["model"] = model_name
        evaluation_rows.append(metrics)
        trained_pipelines[model_name] = pipeline

    summary = pd.DataFrame(evaluation_rows)
    summary = summary.sort_values(
        by=["f1", "recall", "precision", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = str(summary.loc[0, "model"])
    best_pipeline = trained_pipelines[best_model_name]
    best_pipeline.model_name_ = best_model_name
    best_pipeline.feature_schema_ = REQUIRED_MODEL_FEATURES.copy()

    ensure_directory(MODELS_DIR)
    joblib.dump(best_pipeline, MODELS_DIR / "model.pkl")

    # Backward compatibility for existing scripts and deployment code.
    joblib.dump(best_pipeline, MODELS_DIR / "accident_model.pkl")
    joblib.dump(
        best_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl"
    )

    feature_importance_frame = _extract_feature_importance(best_pipeline)
    feature_importance_frame.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    summary.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    report = {
        "best_model": best_model_name,
        "metrics": summary.to_dict(orient="records"),
    }
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Model comparison summary:\n")
    print(summary.to_string(index=False))
    print(f"\nBest model selected: {best_model_name}")
    print(f"Saved pipeline model: {MODELS_DIR / 'model.pkl'}")

    return best_pipeline, summary


if __name__ == "__main__":
    train_and_save_model()
