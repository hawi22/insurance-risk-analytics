import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def prepare_data(df):
    """Encodes categorical data and splits into train/test sets."""
    # Selecting logical features for risk prediction
    features = ['Age', 'Gender', 'Province', 'VehicleType', 'AnnualIncome', 
                'RiskScore', 'TotalPremium', 'AutoMake', 'Deductible']
    
    X = df[features].copy()
    y = df['TotalClaims']
    
    # Label Encoding for categorical columns
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col].astype(str))
        
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_models(X_train, X_test, y_train, y_test):
    """Trains and compares three different models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    metrics = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        metrics.append([name, rmse, r2])
        trained_models[name] = model
        
    return pd.DataFrame(metrics, columns=['Model', 'RMSE', 'R2']), trained_models