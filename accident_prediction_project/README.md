<div align="center">

# SafeRoute AI

### Production-Ready Accident Risk Prediction and Analytics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![GradientBoosting](https://img.shields.io/badge/GradientBoosting-Model-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

SafeRoute AI is an end-to-end machine learning system that predicts the risk of a severe road accident based on road context, vehicle speed, traffic, visibility, and time conditions through an interactive Streamlit app.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Modeling Pipeline](#-modeling-pipeline)
- [Model Evaluation](#-model-evaluation)
- [Run Locally](#-run-locally)
- [Deploy On Streamlit Cloud](#-deploy-on-streamlit-cloud)

---

## Overview

This project upgrades a prototype into a portfolio-quality ML application with:

- Proper training and inference pipeline design
- Model comparison across multiple algorithms
- Input validation and robust preprocessing
- Explainability through feature importance and top risk factors
- Streamlit performance optimizations with caching

---

## Key Features

- Real-time risk prediction with probability and risk category (Low, Medium, High)
- Model benchmark across RandomForestClassifier, LogisticRegression, and GradientBoostingClassifier
- End-to-end sklearn pipeline with preprocessing + model persistence
- Explainability with feature importance and top contributing factors
- Batch CSV prediction upload and downloadable reports
- Prediction history logging and download
- Streamlit performance with cached model/data loading

---

## Tech Stack

| Layer                 | Tools               |
| --------------------- | ------------------- |
| **Dashboard**         | Streamlit, Plotly   |
| **Machine Learning**  | scikit-learn        |
| **Data Processing**   | pandas, numpy       |
| **Model Persistence** | joblib              |
| **Notebook / EDA**    | Jupyter, matplotlib |

---

## Project Structure

```
accident_prediction_project/
│
├── app/
│   ├── app.py                  # New production-style Streamlit app
│   ├── streamlit_app.py        # Deployment entrypoint
│   └── prediction_ui.py        # Legacy UI component
│
├── data/
│   ├── accident_prediction_india.csv          # Main ML training dataset
│   ├── cause-wise-distribution-of-railway-accidents.csv
│   ├── month-of-occurrence-wise-number-of-traffic-accidents.csv
│   ├── road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv
│   └── time-of-occurrence-wise-number-of-traffic-accidents.csv
│
├── models/
│   ├── model.pkl               # Primary persisted sklearn pipeline
│   ├── accident_model.pkl      # Backward-compatible model artifact
│   ├── preprocessor.pkl        # Backward-compatible preprocessor artifact
│   ├── model_comparison.csv    # Evaluation table
│   ├── model_metrics.json      # Metrics and selected model
│   └── feature_importance.csv  # Explainability artifact
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── analytics.py            # Chart generation from supporting datasets
│   ├── data_loader.py          # Defensive multi-format dataset loader
│   ├── evaluate_model.py       # Model evaluation and comparison
│   ├── data_preprocessing.py   # Feature engineering + train/test preparation
│   ├── feature_engineering.py  # Legacy feature helpers
│   ├── predict.py              # Inference + validation + factor explanation
│   ├── preprocessing.py        # Legacy preprocessing helpers
│   ├── train_model.py          # Model training, comparison, persistence
│   └── utils.py                # Shared utilities
│
├── requirements.txt
├── test_quick.py               # Quick validation test suite
└── README.md
```

---

## Modeling Pipeline

The upgraded ML workflow follows production-style design:

1. Load and standardize dataset columns
2. Engineer portfolio-focused features:
   - vehicle_speed
   - weather_condition
   - road_type
   - traffic_density
   - time_of_day
   - driver_fatigue
   - road_lighting
   - visibility_level
3. Train/test split with stratification (where possible)
4. ColumnTransformer preprocessing:
   - Numeric: median imputation + StandardScaler
   - Categorical: constant imputation + OneHotEncoder
5. Train three candidate models:
   - RandomForestClassifier
   - LogisticRegression
   - GradientBoostingClassifier
6. Evaluate using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
7. Persist best model as model.pkl

## Model Evaluation

Model comparisons are exported to:

- models/model_comparison.csv
- models/model_metrics.json
- models/feature_importance.csv

## Run Locally

1. Clone repository

```bash
git clone https://github.com/onimess015/Prototype_accident.git
cd Prototype_accident/accident_prediction_project
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train models and generate artifacts

```bash
python -m src.train_model
```

4. Run Streamlit app

```bash
python start_streamlit.py
```

5. Run verification script

```bash
python run_tests.py
```

## Deploy On Streamlit Cloud

1. Push repository to GitHub
2. In Streamlit Cloud, create a new app from your repo
3. Set main file path to:

```
accident_prediction_project/app/streamlit_app.py
```

4. Ensure root requirements forwarding file exists at repository root:

```
-r accident_prediction_project/requirements.txt
```

5. Ensure runtime pin exists at repository root:

```
python-3.11.9
```

6. Deploy and verify first run creates/loads model artifacts in models/
