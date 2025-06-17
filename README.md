## Insurance Risk Analytics

**An end-to-end machine learning pipeline for smarter, risk-aware insurance pricing - with transparency, explainability, and modularity at its core.**


This repository contains code for analyzing insurance risk using various machine learning techniques. The project focuses on predicting insurance claims and understanding the factors that contribute to risk in the insurance industry.



> **Sprint:** Week 3 | **Client:** AlphaCare Insurance Solutions | **Date:** June 2025

---

## Project Overview

In industries like insurance, risk is currency - and every premium must be explainable, auditable, and fair.

This repository contains the complete pipeline to:

* Predict **claim occurrence** (classification)
* Estimate **claim severity** (regression)
* Combine both to calculate **optimized premiums**
* Visualize and explain predictions with **SHAP values**
* Support actuarial strategy and data-backed underwriting

---

## Key Objectives

| Task                                 | Tools & Models Used                                             |
| ------------------------------------ | --------------------------------------------------------------- |
| 🧼 Data Cleaning & Imputation        | `pandas`, `SimpleImputer`, OneHot/Target Encoding               |
| 🤖 Classification (Claim Occurrence) | `LogisticRegression`, `RandomForest`, `XGBoostClassifier`       |
| 📉 Regression (Claim Amount)         | `LinearRegression`, `RandomForestRegressor`, `XGBoostRegressor` |
| 💸 Premium Optimization              | `P(Claim) × E[ClaimAmount] × LoadingFactor`                     |
| 🔍 Explainability                    | `SHAP` visualizations for model interpretability                |

---


---

## How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**

   ```bash
   python task_4.py
   ```

   This will:

   * Clean and preprocess the data
   * Train all models
   * Optimize premiums
   * Plot model and premium distributions
   * Generate SHAP-based interpretability visuals

---

## Outputs

### Classification Models

* Claim prediction accuracy & F1
* SHAP values to interpret key claim drivers

### Regression Models

* RMSE and R² for claim severity
* Explainable SHAP plots for regression

### Premiums

* Risk-adjusted premiums calculated via:

  ```
  Optimized Premium = P(Claim) × E[ClaimAmount] × LoadingFactor
  ```
* Visualized premium distribution with clear outlier detection

---


## Versioning & Reproducibility

* ✅ Git versioning enabled
* ✅ DVC-compatible for dataset version control
* ✅ Modular Python codebase

---

## Requirements

```txt
pandas
numpy
scikit-learn
xgboost
matplotlib
shap
category_encoders
```

---




