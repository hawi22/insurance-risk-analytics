import pandas as pd
import numpy as np
from scipy import stats

def perform_chi2_test(df, feature, target='Claimed'):
    """Used for Claim Frequency (Categorical vs Categorical)."""
    contingency_table = pd.crosstab(df[feature], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p

def perform_ttest(group_a, group_b):
    """Used for Claim Severity or Margin (Numerical vs Categorical)."""
    t_stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
    return t_stat, p