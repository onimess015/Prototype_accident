<div align="center">

# 🛣️ SafeRoute AI

### Road Accident Risk Prediction & Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**SafeRoute AI** is an end-to-end machine learning system that predicts the risk of a severe road accident based on driver behavior, vehicle type, road conditions, and environmental factors — all visualized through an interactive Streamlit dashboard.

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
- [Model Performance](#-model-performance)
- [Dashboard Tabs](#-dashboard-tabs)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

Road accidents are driven by a complex mix of human, mechanical, and environmental factors. **SafeRoute AI** takes 12 input variables and returns a real-time risk score (Low / Medium / High) alongside visual analytics drawn from national traffic datasets.

> Built as a prototype for intelligent road safety analysis using real-world Indian accident data.

---

## ✨ Key Features

- 🎯 **Real-time Risk Prediction** — Instant accident risk scoring with confidence percentage
- 📊 **Interactive Analytics Dashboard** — 5-tab Streamlit interface with Plotly charts
- 🧠 **Multi-model Training** — Logistic Regression, Random Forest, and XGBoost compared automatically
- 🔧 **Modular Architecture** — Clean separation of data loading, preprocessing, training, and inference
- 📈 **Trend Analysis** — Cause-wise, road-type, time-of-day, and monthly accident breakdowns
- ⚡ **Efficient Inference** — Cached model loading, ~47ms prediction latency

---

## 🛠️ Tech Stack

| Layer                 | Tools                 |
| --------------------- | --------------------- |
| **Dashboard**         | Streamlit, Plotly     |
| **Machine Learning**  | scikit-learn, XGBoost |
| **Data Processing**   | pandas, numpy         |
| **Model Persistence** | joblib                |
| **Notebook / EDA**    | Jupyter, matplotlib   |

---

## 📁 Project Structure

```
accident_prediction_project/
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit dashboard
│   └── prediction_ui.py        # Professional prediction result UI
│
├── data/
│   ├── accident_prediction_india.csv          # Main ML training dataset
│   ├── cause-wise-distribution-of-railway-accidents.csv
│   ├── month-of-occurrence-wise-number-of-traffic-accidents.csv
│   ├── road-classification-wise-number-of-road-accidents-injuries-and-deaths.csv
│   └── time-of-occurrence-wise-number-of-traffic-accidents.csv
│
├── models/
│   ├── accident_model.pkl      # Trained best model
│   └── preprocessor.pkl        # Fitted preprocessing pipeline
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── src/
│   ├── analytics.py            # Chart generation from supporting datasets
│   ├── data_loader.py          # Defensive multi-format dataset loader
│   ├── evaluate_model.py       # Model evaluation and comparison
│   ├── feature_engineering.py  # Feature selection and target encoding
│   ├── predict.py              # Inference pipeline
│   ├── preprocessing.py        # ColumnTransformer pipeline
│   ├── train_model.py          # Main training script
│   └── utils.py                # Shared utilities
│
├── requirements.txt
├── test_quick.py               # Quick validation test suite
└── README.md
```

---

## 📂 Datasets

### Training Dataset

| File                            | Purpose                                      |
| ------------------------------- | -------------------------------------------- |
| `accident_prediction_india.csv` | 3,000-row dataset used for ML model training |

### Analytics Datasets (Dashboard Only)

| File                                               | Used For                 |
| -------------------------------------------------- | ------------------------ |
| `cause-wise-distribution-of-railway-accidents.csv` | Cause analysis charts    |
| `road-classification-wise-...csv`                  | Road type breakdown      |
| `time-of-occurrence-wise-...csv`                   | Time-of-day trend charts |
| `month-of-occurrence-wise-...csv`                  | Monthly trend charts     |

> The data loader auto-detects available file variants (CSV/XLSX) and handles missing columns gracefully.

---

## ⚙️ How It Works

```
Raw Input (12 features)
        ↓
  Data Cleaning & Standardization
        ↓
  Feature Engineering (binary target, encoding)
        ↓
  ColumnTransformer (StandardScaler + OneHotEncoder)
        ↓
  Model Selection (Logistic Regression / Random Forest / XGBoost)
        ↓
  Best Model Saved → Inference Pipeline
        ↓
  Risk Score Output (Low / Medium / High) + Recommendations
```

**Model Selection Logic:** Models are ranked by F1 score first, then ROC AUC. The best-performing model is saved automatically to `models/`.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/onimess015/Prototype_accident.git
cd Prototype_accident/accident_prediction_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python -m src.train_model
```

### 4. Launch the Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

### 5. Run Tests

```bash
python test_quick.py
```

---

## 📊 Model Performance

| Model                             | Accuracy | F1 Score |
| --------------------------------- | -------- | -------- |
| **Random Forest** ✅ _(selected)_ | 65.33%   | 0.7903   |
| XGBoost                           | ~58%     | —        |
| Logistic Regression               | —        | —        |

> The dataset has a class imbalance of ~1.9:1 (Severe vs Low Risk). `class_weight="balanced"` is applied during training.

---

## 🖥️ Dashboard Tabs

| Tab                       | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| **🎯 Prediction**         | Enter driver/vehicle/road details and get an instant risk score |
| **⚠️ Accident Causes**    | Bar chart of cause-wise accident distribution                   |
| **🛣️ Road Type Analysis** | Breakdown of accidents by road classification                   |
| **🕐 Time Trends**        | Hourly/time-of-day accident frequency chart                     |
| **📅 Monthly Trends**     | Month-wise accident occurrence patterns                         |

---

## 🔮 Future Improvements

- [ ] SHAP-based model explainability for individual predictions
- [ ] Live weather and traffic feed integration
- [ ] Model drift monitoring and automated retraining
- [ ] FastAPI backend for production REST API serving
- [ ] Downloadable PDF risk reports
- [ ] Map-based geospatial accident hotspot visualization
