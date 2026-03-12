import joblib

m = joblib.load("models/accident_model.pkl")
print(f"Model type: {type(m).__name__}")
print(f'Model_name_: {getattr(m, "model_name_", "unknown")}')
print(f'Has predict: {hasattr(m, "predict")}')
print(f'Has fit: {hasattr(m, "fit")}')
