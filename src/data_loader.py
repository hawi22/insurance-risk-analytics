import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads the insurance dataset using detected columns."""
   
    df = pd.read_csv(file_path, sep=',')
    
    
    if 'TransactionDate' in df.columns:
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    return df

def clean_data(df):
    """Basic cleaning."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
            
    return df

def calculate_metrics(df):
    """Calculates Loss Ratio and Margin."""
    
    df['LossRatio'] = np.where(df['TotalPremium'] != 0, 
                               df['TotalClaims'] / df['TotalPremium'], 0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df