import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report

def evaluate_regression(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        }
    return results

def evaluate_classification(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "Report": classification_report(y_test, preds, digits=3)
        }
    return results