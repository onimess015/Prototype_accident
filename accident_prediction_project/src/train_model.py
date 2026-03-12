from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from .data_loader import load_main_dataset
from .evaluate_model import compare_models, evaluate_classifier
from .feature_engineering import select_features_and_target
from .preprocessing import clean_dataframe, fit_preprocessor, transform_features
from .utils import MODELS_DIR, ensure_directory


def train_and_save_model() -> tuple[object, object]:
    raw_dataframe = load_main_dataset()
    cleaned_dataframe = clean_dataframe(raw_dataframe)
    X, y = select_features_and_target(cleaned_dataframe)

    stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    preprocessor = fit_preprocessor(X_train)
    preprocessor.feature_schema_ = list(X_train.columns)
    X_train_transformed = transform_features(preprocessor, X_train)
    X_test_transformed = transform_features(preprocessor, X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
        "RandomForest": RandomForestClassifier(
            random_state=42, n_estimators=300, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_transformed, y_train)
        trained_models[model_name] = model
        metrics = evaluate_classifier(model, X_test_transformed, y_test)
        results[model_name] = metrics

    comparison = compare_models(results)
    best_model_name = comparison.loc[0, "model"]
    best_model = trained_models[best_model_name]
    best_model.model_name_ = best_model_name

    ensure_directory(MODELS_DIR)
    joblib.dump(best_model, MODELS_DIR / "accident_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    print(f"\nBest model selected: {best_model_name}")
    print(f"Saved model to: {MODELS_DIR / 'accident_model.pkl'}")
    print(f"Saved preprocessor to: {MODELS_DIR / 'preprocessor.pkl'}")
    return best_model, preprocessor


if __name__ == "__main__":
    train_and_save_model()
