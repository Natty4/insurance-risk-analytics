import json
import pandas as pd

import sys
import pathlib

# Dynamically get the root directory (project root where `src/` is located)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR

import sys
# print("\n".join(sys.path))

# Add src/ to sys.path
sys.path.append(str(SRC_DIR))


# print("\n".join(sys.path), '---->')

from utils.cleaning_and_imputation_functions import preprocess_data

# File paths
input_file = "data/raw/insurance_data.csv"
output_file = "data/processed/insurance_data_cleaned.csv"
metrics_file = "metrics/summary.json"

def main():
    # Run the full cleaning pipeline
    df = preprocess_data(input_file, output_file)

    # If cleaning succeeded, generate summary metrics
    if df is not None:
        metrics = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isnull().sum().sum()),
            "mean_loss_ratio": float(df['LossRatio'].mean()),
            "writtenoff_ratio": float((df['WrittenOff'] == 'Yes').mean())
        }

        # Save metrics to JSON
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        print("✅ Cleaning and metric generation complete.")
    else:
        print("❌ Cleaning failed.")

if __name__ == "__main__":
    main()