import pandas as pd
from sklearn.impute import SimpleImputer

def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    df[cat_cols] = df[cat_cols].fillna('Missing')
    return df