# SafeRoute AI

## Road Accident Risk Prediction and Analytics Dashboard

SafeRoute AI is a Streamlit-based machine learning application that predicts road accident risk using driver, vehicle, road, and environmental context while also providing accident trend analytics from supporting datasets.

## Problem Statement

Road accidents are influenced by a combination of driver behavior, road characteristics, environmental conditions, and vehicle factors. This project predicts the probability of a severe accident and presents supporting accident analytics in an interactive dashboard.

## Datasets Used

### Main dataset for ML training only

- `data/accident_prediction_india.xls` or `data/accident_prediction_india.csv`

### Supporting datasets for analytics only

- `data/cause-wise-distribution-of-road-accidents-and-unmanned-railway-crossing-accidents.csv`
- `data/cause-wise-distribution-of-railway-accidents.csv`
- `data/road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv`
- `data/time-of-occurrence-wise-number-of-traffic-accidents.csv`
- `data/month-of-occurrence-wise-number-of-traffic-accidents.csv`

The loader is defensive and auto-detects the file variants that are present in the workspace. Only the main accident prediction dataset is used for training. The aggregated supporting datasets are used only for dashboard analytics.

## Architecture Summary

The application follows a modular architecture:

- Streamlit UI for prediction inputs and dashboard pages
- Application logic for validation and feature preparation
- Preprocessing pipeline using `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
- Model training and selection across Logistic Regression, Random Forest, and XGBoost
- Saved model artifacts for inference consistency
- Analytics module for charting cause, road type, time, and month trends

## Folder Structure

```text
accident_prediction_project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── accident_prediction_india.csv
│   ├── cause-wise-distribution-of-railway-accidents.csv
│   ├── month-of-occurrence-wise-number-of-traffic-accidents.csv
│   ├── road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv
│   └── time-of-occurrence-wise-number-of-traffic-accidents.csv
├── models/
│   ├── accident_model.pkl
│   └── preprocessor.pkl
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── analytics.py
│   ├── data_loader.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Model Training Workflow

1. Load the main dataset.
2. Standardize column names and clean duplicate or malformed rows.
3. Select the best available features and map them to a stable schema.
4. Convert accident severity into a binary target.
5. Split train and test data.
6. Fit the preprocessor on training features only.
7. Train Logistic Regression, Random Forest, and XGBoost.
8. Compare models using F1 score first, then ROC AUC.
9. Save the best model and the fitted preprocessor in `models/`.

## How to Run the Project

From the `accident_prediction_project` folder:

1. `pip install -r requirements.txt`
2. `python -m src.train_model`
3. `streamlit run app/streamlit_app.py`

## System Design Summary

SafeRoute AI is a **Streamlit-based machine learning application that predicts road accident risk using a preprocessing and prediction pipeline while providing accident trend analytics through integrated datasets.**

## Future Improvements

- Add SHAP-based model explainability for individual predictions
- Integrate live weather and traffic feeds
- Add model monitoring and drift checks
- Introduce API deployment with FastAPI for production serving
- Add downloadable risk reports and dashboard exports
