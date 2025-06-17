import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def compute_claim_metrics(df):
    df['ClaimOccurred'] = df['TotalClaims'] > 0
    df['ClaimFrequency'] = df.groupby('PolicyID')['ClaimOccurred'].transform('max')
    df['ClaimSeverity'] = df['TotalClaims'].where(df['ClaimOccurred'], 0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def t_test_groups(df, group_col, metric_col, group_a, group_b):
    group1 = df[df[group_col] == group_a][metric_col].dropna()
    group2 = df[df[group_col] == group_b][metric_col].dropna()
    stat, p = ttest_ind(group1, group2, equal_var=False)
    return {
        "metric": metric_col,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "t_stat": stat,
        "p_value": p,
        "significant": p < 0.05
    }

def chi_squared_test(df, group_col, target_col):
    contingency = pd.crosstab(df[group_col], df[target_col])
    chi2, p, _, _ = chi2_contingency(contingency)
    return {
        "group_col": group_col,
        "target_col": target_col,
        "chi2": chi2,
        "p_value": p,
        "significant": p < 0.05
    }