# 📞 Telco Customer Churn Prediction

> An end-to-end machine learning project — from raw telecom data to a deployed Streamlit web app, built to showcase ML and Data Science skills.

---

## 🎯 Project Overview

Customer churn — when a subscriber leaves a telecom service — directly impacts revenue. This project builds a full churn prediction pipeline on the **IBM Telco Customer Churn dataset** (7,043 customers), tackling a real-world imbalanced classification problem with ~26% churn rate.

The goal: identify which customers are likely to leave, and why.

---

## 🧩 Project Components

| File | Description |
|---|---|
| `Telco_churn.ipynb` | Core ML pipeline — EDA, preprocessing, modeling, tuning |
| `Streamlit.ipynb` | Interactive web app for real-time predictions |
| `Telco_visualization.pbix` | Power BI dashboard for business reporting |

---

## 📊 Dataset at a Glance

- **7,043 customers**, 20 features (demographic, subscription, billing)
- **Churn rate:** ~26% (imbalanced)
- **11 missing values** in `TotalCharges` — caught and imputed
- Key features: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`

---

## 🔄 ML Pipeline Walkthrough

### 1. Exploratory Data Analysis
- Pie charts for gender, partner status, and contract type distribution
- Correlation heatmap across numerical features
- `seaborn.pairplot()` to visualize churn clusters across `tenure`, `MonthlyCharges`, and `TotalCharges`

**Key finding:** Month-to-month contract customers (55% of the dataset) churn at significantly higher rates than one-year or two-year subscribers.

---

### 2. Preprocessing

Smart encoding chosen per column type:

| Strategy | Applied To |
|---|---|
| `BinaryEncoder` | `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` |
| `TargetEncoder` | `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`, and 6 others |
| `StandardScaler` | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` |
| `SimpleImputer (mean)` | `TotalCharges` (11 nulls) |

---

### 3. Handling Class Imbalance — SMOTE

Applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively on the training set to generate synthetic churn samples, preventing data leakage into test evaluation.

---

### 4. Baseline Model

Initial pipeline with `RandomForestClassifier` + SMOTE:

```
              precision    recall    f1-score
    No Churn    0.84        0.80        0.82
       Churn    0.52        0.58        0.55
    accuracy                            0.74
```

Confusion matrix: `[[1227, 312], [240, 334]]` — decent recall on churn but room to improve.

---

### 5. Hyperparameter Tuning with Optuna

Ran **50 Optuna trials** using `MedianPruner` (n_startup_trials=5, n_warmup_steps=1) to tune all three models simultaneously.

**Best trial:** Trial 36 — Score: **0.8567**

Best hyperparameters found:
- RF: `n_estimators=400`, `max_depth=19`
- XGB: `n_estimators=350`, `learning_rate=0.0394`, `max_depth=9`, `subsample=0.586`, `colsample_bytree=0.968`
- LGBM: `n_estimators=250`, `learning_rate=0.061`, `max_depth=4`, `num_leaves=20`

---

### 6. Ensemble — VotingClassifier

Combined three models into a soft **VotingClassifier**:
- `RandomForestClassifier`
- `XGBClassifier`
- `LGBMClassifier`
- `LogisticRegression` (as a linear anchor)

Post-SMOTE VotingClassifier results:

```
              precision    recall    f1-score
    No Churn    0.74        0.98        0.84      (1539)
       Churn    0.97        0.65        0.78      (1539)
    accuracy                            0.82
   macro avg    0.85        0.82        0.81
```

Confusion matrix: `[[1504, 35], [531, 1008]]`

---

### 7. Feature Selection — SelectFromModel

Used `SelectFromModel` with a Random Forest base and `threshold='median'` to reduce dimensionality.

**12 features selected:**

`tenure`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

Final model after feature selection:

```
              precision    recall    f1-score
    No Churn    0.75        0.96        0.84      (1539)
       Churn    0.95        0.67        0.79      (1539)
    accuracy                            0.82
   macro avg    0.85        0.82        0.81
```

Confusion matrix: `[[1485, 54], [506, 1033]]`

Performance held stable with fewer features — cleaner, more generalizable model.

---

## 🚀 Streamlit Deployment

The trained model was serialized with `joblib` and served through a Streamlit web app for real-time predictions — no code needed by the user.

**Saved artifacts:**

| File | Contents |
|---|---|
| `trained_model.pkl` | Final VotingClassifier |
| `T_encoder.pkl` | TargetEncoder |
| `B_encoder.pkl` | BinaryEncoder |
| `S_scaler.pkl` | StandardScaler |
| `selected_columns.pkl` | 12 selected features |

---

## 📈 Power BI Dashboard

`Telco_visualization.pbix` provides an interactive business-facing dashboard to explore churn patterns across demographics, services, and contract types — without touching any code.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML | `scikit-learn`, `xgboost`, `lightgbm` |
| Encoding | `category_encoders` (BinaryEncoder, TargetEncoder) |
| Imbalance | `imbalanced-learn` (SMOTE) |
| Tuning | `optuna` |
| Deployment | `streamlit`, `joblib` |
| BI | Microsoft Power BI |

---

## 📁 Project Structure

```
Telco-Churn-Prediction/
│
├── Telco_churn.ipynb          # Core ML pipeline
├── Streamlit.ipynb            # Streamlit prediction app
├── Telco_visualization.pbix   # Power BI dashboard
│
├── trained_model.pkl
├── T_encoder.pkl
├── B_encoder.pkl
├── S_scaler.pkl
├── selected_columns.pkl
│
└── README.md
```

---

## 📌 Results Summary

| Stage | Accuracy | Macro F1 | Churn Precision | Churn Recall |
|---|---|---|---|---|
| Baseline (RF + SMOTE) | 0.74 | 0.68 | 0.52 | 0.58 |
| Optuna Best Trial Score | — | **0.857** | — | — |
| Voting Classifier (post-SMOTE) | **0.82** | **0.81** | **0.97** | 0.65 |
| After Feature Selection | **0.82** | **0.81** | **0.95** | 0.67 |

> High churn precision (0.95–0.97) means when the model flags a customer as likely to churn, it's almost always correct — critical for targeting retention campaigns efficiently.

---

*Dataset: [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)*