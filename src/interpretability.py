import shap
import matplotlib.pyplot as plt

def explain_model_with_shap(model, X_sample, feature_names=None, max_display=10):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=max_display)