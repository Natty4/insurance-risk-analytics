import pandas as pd
import numpy as np
import json, sys, pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR
sys.path.append(str(SRC_DIR))

from src.utils.hypothesis_testing import compute_claim_metrics, t_test_groups, chi_squared_test

df = pd.read_csv("data/processed/insurance_data_cleaned.csv")

# Compute risk and margin metrics
df = compute_claim_metrics(df)

results = []


provinces = df['Province'].dropna().unique()
results.append(t_test_groups(df, 'Province', 'ClaimFrequency', provinces[0], provinces[1]))

top_zips = df['PostalCode'].value_counts().nlargest(2).index.tolist()
results.append(t_test_groups(df, 'PostalCode', 'ClaimFrequency', top_zips[0], top_zips[1]))

results.append(t_test_groups(df, 'PostalCode', 'Margin', top_zips[0], top_zips[1]))

results.append(t_test_groups(df, 'Gender', 'ClaimFrequency', 'Male', 'Female'))

results.append(chi_squared_test(df, 'Gender', 'ClaimOccurred'))

def convert_numpy(obj):
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Save results
with open("metrics/hypothesis_test_results.json", "w") as f:
    print("Saving results to metrics/hypothesis_test_results.json ...")
    json.dump(results, f, indent=4, default=convert_numpy)

print("✅ Hypothesis testing completed. Results saved to metrics/hypothesis_test_results.json.")