import pandas as pd
import matplotlib.pyplot as plt
import sys, pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR
sys.path.append(str(SRC_DIR))

from src.cleaning import clean_and_impute
from src.preprocessing import prepare_severity_data, prepare_classification_data
from src.models import train_regression_models, train_classification_models
from src.evaluation import evaluate_regression, evaluate_classification
from src.optimization import compute_optimized_premium
from src.interpretability import explain_model_with_shap

#
DATA_PATH = "data/processed/insurance_data_cleaned.csv"
df = pd.read_csv(DATA_PATH)


df_clean = clean_and_impute(df)


high_card_cols = ["PolicyID", "PostalCode", "Model", "make"]



(X_train_sev, X_test_sev, y_train_sev, y_test_sev), df_sev = prepare_severity_data(df_clean, high_card_cols)


X_train_cls, X_test_cls, y_train_cls, y_test_cls = prepare_classification_data(df_clean, high_card_cols)


reg_models = train_regression_models(X_train_sev, y_train_sev)
cls_models = train_classification_models(X_train_cls, y_train_cls)


reg_results = evaluate_regression(reg_models, X_test_sev, y_test_sev)
cls_results = evaluate_classification(cls_models, X_test_cls, y_test_cls)



n_samples = min(X_test_cls.shape[0], X_test_sev.shape[0])
X_cls_aligned = X_test_cls[:n_samples]
X_sev_aligned = X_test_sev[:n_samples]


p_claim = cls_models["XGBoost"].predict_proba(X_cls_aligned)[:, 1]
expected_claim = reg_models["RandomForest"].predict(X_sev_aligned)


optimized_premium = compute_optimized_premium(p_claim, expected_claim, loading_factor=1.2)


plt.figure(figsize=(8, 4))
plt.hist(optimized_premium, bins=40, color='dodgerblue', edgecolor='black')
plt.title("📈 Risk-Based Premium Distribution")
plt.xlabel("Estimated Premium (Rand)")
plt.ylabel("Number of Policies")
plt.grid(True)
plt.tight_layout()
plt.show()


print("🔹 Regression Model Results:")
for name, res in reg_results.items():
    print(f"{name} → RMSE: {res['RMSE']:.2f}, R²: {res['R2']:.3f}")

print("\n🔸 Classification Model Results:")
for name, res in cls_results.items():
    print(f"{name} → Accuracy: {res['Accuracy']:.3f}, F1: {res['F1']:.3f}")



print("\n🧠 SHAP Summary: Claim Severity (RandomForest)")
X_sample_reg = X_train_sev[:500]
explain_model_with_shap(reg_models["RandomForest"], X_sample_reg)


print("\n🧠 SHAP Summary: Claim Probability (XGBoost)")
X_sample_cls = X_train_cls[:500]
explain_model_with_shap(cls_models["XGBoost"], X_sample_cls)


import numpy as np


cls_df = pd.DataFrame(cls_results).T.reset_index().rename(columns={"index": "Model"})
cls_df[["Accuracy", "F1"]] = cls_df[["Accuracy", "F1"]].astype(float)

cls_df.plot(x="Model", y=["Accuracy", "F1"], kind="bar", figsize=(8, 4), colormap="viridis")
plt.title("🔸 Classification Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


reg_df = pd.DataFrame(reg_results).T.reset_index().rename(columns={"index": "Model"})

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
reg_df.plot(x="Model", y="RMSE", kind="bar", ax=ax[0], color='salmon', legend=False)
reg_df.plot(x="Model", y="R2", kind="bar", ax=ax[1], color='seagreen', legend=False)

ax[0].set_title("RMSE Comparison (Lower is Better)")
ax[1].set_title("R² Comparison (Higher is Better)")
for a in ax:
    a.set_xlabel("Model")
    a.grid(True, axis='y')

plt.tight_layout()
plt.show()