from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, X_test, y_test) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_score = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if y_score is not None else None,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    figure, axis = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(metrics["confusion_matrix"]).plot(ax=axis, colorbar=False)
    axis.set_title(f"Confusion Matrix: {model.__class__.__name__}")
    plt.close(figure)
    metrics["confusion_matrix_figure"] = figure

    print(metrics["classification_report"])
    return metrics


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, model_metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": model_metrics["accuracy"],
                "precision": model_metrics["precision"],
                "recall": model_metrics["recall"],
                "f1": model_metrics["f1"],
                "roc_auc": (
                    model_metrics["roc_auc"]
                    if model_metrics["roc_auc"] is not None
                    else -1
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["f1", "roc_auc", "accuracy"], ascending=False
    ).reset_index(drop=True)
    print("\nModel comparison summary:\n")
    print(summary.to_string(index=False))
    return summary
