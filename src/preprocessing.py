import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_col, high_card_cols):
    df = df.drop(columns=high_card_cols, errors='ignore')
    y = df[target_col]
    X = df.drop(columns=[target_col])
    encoder = OneHotEncoder(use_cat_names=True)
    X_encoded = encoder.fit_transform(X)
    return X_encoded, y

def prepare_severity_data(df, high_card_cols, target_col="TotalClaims", test_size=0.3, random_state=42):
    df_sev = df[df["TotalClaims"] > 0].copy()
    X, y = preprocess_data(df_sev, target_col=target_col, high_card_cols=high_card_cols)
    return train_test_split(X, y, test_size=test_size, random_state=random_state), df_sev

def prepare_classification_data(df, high_card_cols, target_col="HasClaim", test_size=0.3, random_state=42):
    df_cls = df.copy()
    df_cls[target_col] = (df_cls["TotalClaims"] > 0).astype(int)
    X, y = preprocess_data(df_cls, target_col=target_col, high_card_cols=high_card_cols)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)