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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
    model: Pipeline,
    X_data: pd.DataFrame,
    y_data: pd.Series,
    threshold: float = 0.5,
) -> dict[str, object]:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_data)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_score = None
        y_pred = model.predict(X_data)

    matrix = confusion_matrix(y_data, y_pred)
    if matrix.shape == (2, 2):
        tn, fp, fn, tp = matrix.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_data, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_data, y_pred)),
        "precision": float(precision_score(y_data, y_pred, zero_division=0)),
        "recall": float(recall_score(y_data, y_pred, zero_division=0)),
        "f1": float(f1_score(y_data, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_data, y_pred)),
        "specificity": specificity,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": (
            float(roc_auc_score(y_data, y_score))
            if y_score is not None and len(np.unique(y_data)) > 1
            else None
        ),
    }
    return metrics


def _find_best_threshold(y_true: pd.Series, y_scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.20, 0.81, 0.02):
        y_pred = (y_scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


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


def _load_previous_best_metrics() -> dict[str, object] | None:
    metrics_path = MODELS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return None

    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    rows = report.get("metrics", [])
    if not rows:
        return None
    return rows[0]


def _write_training_improvement_report(
    previous_best: dict[str, object] | None,
    new_best: dict[str, object],
) -> None:
    keys = ["accuracy", "balanced_accuracy", "f1", "mcc", "specificity", "roc_auc"]
    deltas: dict[str, float | None] = {}
    for key in keys:
        if previous_best is None:
            deltas[key] = None
            continue
        old_value = previous_best.get(key)
        new_value = new_best.get(key)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            deltas[key] = float(new_value - old_value)
        else:
            deltas[key] = None

    report = {
        "previous_model": previous_best.get("model") if previous_best else None,
        "new_model": new_best.get("model"),
        "previous": previous_best,
        "current": new_best,
        "delta": deltas,
    }

    (MODELS_DIR / "training_improvement_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Training Improvement Report",
        "",
        f"- Previous best model: {report['previous_model']}",
        f"- New best model: {report['new_model']}",
        "",
        "## Metric Delta (new - previous)",
    ]
    for key in keys:
        delta_value = deltas[key]
        if delta_value is None:
            lines.append(f"- {key}: n/a")
        else:
            lines.append(f"- {key}: {delta_value:+.6f}")

    (MODELS_DIR / "training_improvement_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def train_and_save_model() -> tuple[Pipeline, pd.DataFrame]:
    previous_best = _load_previous_best_metrics()
    prepared = load_and_prepare_data(test_size=0.2, holdout_size=0.1, random_state=42)
    X_train, X_test, X_holdout = prepared.X_train, prepared.X_test, prepared.X_holdout
    y_train, y_test, y_holdout = prepared.y_train, prepared.y_test, prepared.y_holdout

    numeric_features = ["vehicle_speed", "driver_age"]
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
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
            class_weight="balanced",
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "balanced_accuracy": "balanced_accuracy",
                "f1": "f1",
                "roc_auc": "roc_auc",
                "accuracy": "accuracy",
            },
            n_jobs=-1,
            return_train_score=False,
        )

        X_tune, X_val, y_tune, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train if y_train.nunique() > 1 else None,
        )

        pipeline.fit(X_tune, y_tune)
        if hasattr(pipeline, "predict_proba"):
            val_scores = pipeline.predict_proba(X_val)[:, 1]
            threshold = _find_best_threshold(y_val, val_scores)
        else:
            threshold = 0.5

        pipeline.fit(X_train, y_train)
        pipeline.best_threshold_ = threshold

        train_metrics = _evaluate_model(pipeline, X_train, y_train, threshold=threshold)
        test_metrics = _evaluate_model(pipeline, X_test, y_test, threshold=threshold)

        metrics = {
            "model": model_name,
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "mcc": test_metrics["mcc"],
            "specificity": test_metrics["specificity"],
            "roc_auc": test_metrics["roc_auc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "test_f1": test_metrics["f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "accuracy_gap": float(train_metrics["accuracy"] - test_metrics["accuracy"]),
            "f1_gap": float(train_metrics["f1"] - test_metrics["f1"]),
            "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
            "cv_balanced_accuracy_mean": float(
                np.mean(cv_results["test_balanced_accuracy"])
            ),
            "cv_balanced_accuracy_std": float(
                np.std(cv_results["test_balanced_accuracy"])
            ),
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "decision_threshold": threshold,
            "overfitting_flag": bool(
                (train_metrics["accuracy"] - test_metrics["accuracy"]) > 0.10
                or (train_metrics["f1"] - test_metrics["f1"]) > 0.10
            ),
            "degenerate_prediction_flag": bool(
                test_metrics["specificity"] == 0.0 or test_metrics["recall"] == 0.0
            ),
        }
        evaluation_rows.append(metrics)
        trained_pipelines[model_name] = pipeline

    summary = pd.DataFrame(evaluation_rows)
    summary = summary.sort_values(
        by=[
            "balanced_accuracy",
            "mcc",
            "f1",
            "specificity",
            "accuracy",
            "cv_balanced_accuracy_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = str(summary.loc[0, "model"])
    best_pipeline = trained_pipelines[best_model_name]
    best_pipeline.model_name_ = best_model_name
    best_pipeline.feature_schema_ = REQUIRED_MODEL_FEATURES.copy()

    holdout_threshold = float(getattr(best_pipeline, "best_threshold_", 0.5))
    holdout_metrics = _evaluate_model(
        best_pipeline,
        X_holdout,
        y_holdout,
        threshold=holdout_threshold,
    )
    for metric_name, metric_value in holdout_metrics.items():
        summary.loc[summary["model"] == best_model_name, f"holdout_{metric_name}"] = (
            metric_value
        )

    summary.loc[summary["model"] == best_model_name, "holdout_threshold"] = (
        holdout_threshold
    )

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
        "data_quality_report": prepared.quality_report,
    }
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Model comparison summary:\n")
    print(summary.to_string(index=False))
    print(f"\nBest model selected: {best_model_name}")
    print(f"Saved pipeline model: {MODELS_DIR / 'model.pkl'}")

    _write_training_improvement_report(
        previous_best=previous_best,
        new_best=summary.loc[0].to_dict(),
    )

    return best_pipeline, summary


if __name__ == "__main__":
    train_and_save_model()
