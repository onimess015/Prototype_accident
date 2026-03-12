"""Check if optimized training completed successfully."""

import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import MODELS_DIR
from data_loader import load_main_dataset
from preprocessing import clean_dataframe, fit_preprocessor, transform_features
from feature_engineering import select_features_and_target
from train_model_optimized import create_interaction_features
from sklearn.model_selection import train_test_split

# Check if model files exist
model_path = MODELS_DIR / "accident_model.pkl"
preprocessor_path = MODELS_DIR / "preprocessor.pkl"

print(f"Checking model files...")
print(f"  Model exists: {model_path.exists()}")
print(f"  Preprocessor exists: {preprocessor_path.exists()}")

if model_path.exists() and preprocessor_path.exists():
    # Load and test
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        print(f"\n[OK] Models loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model name: {getattr(model, 'model_name_', 'unknown')}")

        # Try a quick prediction
        raw_df = load_main_dataset()[:100]
        cleaned_df = clean_dataframe(raw_df)
        X, y = select_features_and_target(cleaned_df)
        X_enhanced = create_interaction_features(X)
        X_transformed = preprocessor.transform(X_enhanced)

        predictions = model.predict(X_transformed)
        accuracy = (predictions == y.values).mean()

        print(f"\n[OK] Test prediction successful")
        print(f"  Accuracy on sample: {accuracy:.2%}")
        print(f"\n[SUCCESS] Optimized training appears to have completed!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
else:
    print(f"\n[ERROR] Model files not found")
    print(f"  Model path: {model_path}")
    print(f"  Preprocessor path: {preprocessor_path}")
