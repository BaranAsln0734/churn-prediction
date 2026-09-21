import pytest
import pandas as pd
import numpy as np
import os
import joblib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_total_charges, engineer_features, transform_input_dataframe

def test_clean_total_charges():
    df = pd.DataFrame({
        'TotalCharges': [' ', '100.5', '250.0', np.nan]
    })
    cleaned = clean_total_charges(df)
    assert cleaned['TotalCharges'].iloc[0] == 0.0
    assert cleaned['TotalCharges'].iloc[1] == 100.5
    assert cleaned['TotalCharges'].iloc[3] == 0.0

def test_engineer_features():
    df = pd.DataFrame({
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No'],
        'OnlineSecurity': ['Yes', 'No'],
        'OnlineBackup': ['No', 'No'],
        'DeviceProtection': ['No', 'No'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No'],
        'tenure': [3, 24],
        'MonthlyCharges': [50.0, 70.0],
        'TotalCharges': [150.0, 1680.0]
    })
    feat_df = engineer_features(df)
    assert feat_df['NumServices'].iloc[0] == 3
    assert feat_df['NumServices'].iloc[1] == 0
    assert feat_df['IsNewCustomer'].iloc[0] == 1
    assert feat_df['IsNewCustomer'].iloc[1] == 0
    assert abs(feat_df['AvgMonthlySpend'].iloc[0] - 50.0) < 1e-4

def test_transform_input_dataframe():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encoders_path = os.path.join(base_dir, 'models', 'encoders.pkl')
    feature_names_path = os.path.join(base_dir, 'models', 'feature_names.pkl')
    
    if not os.path.exists(encoders_path) or not os.path.exists(feature_names_path):
        pytest.skip("Model artifacts missing, skipping.")
        
    encoders = joblib.load(encoders_path)
    feature_names = joblib.load(feature_names_path)
    
    df = pd.DataFrame([{
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 45.0,
        'TotalCharges': 540.0
    }])
    
    transformed = transform_input_dataframe(df, encoders, feature_names)
    assert list(transformed.columns) == feature_names
    assert len(transformed) == 1
    assert not transformed.isna().any().any()
