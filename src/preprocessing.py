import os
import pandas as pd
import numpy as np

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans whitespace in TotalCharges, converts to float, and fills empty tenure with 0.0."""
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers NumServices, AvgMonthlySpend, and IsNewCustomer features."""
    service_cols = [
        c for c in [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ] if c in df.columns
    ]
    if service_cols:
        df['NumServices'] = df[service_cols].apply(lambda row: sum(str(x).lower() == 'yes' for x in row), axis=1)
    elif 'NumServices' not in df.columns:
        df['NumServices'] = 0

    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['AvgMonthlySpend'] = np.where(df['tenure'] == 0, df['MonthlyCharges'], df['TotalCharges'] / df['tenure'].replace(0, 1))
        df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)

    return df

def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encodes categorical columns safely using pre-fitted LabelEncoder dictionary."""
    for col, enc in encoders.items():
        if col in df.columns:
            known_classes = set(enc.classes_)
            df[col] = df[col].astype(str).map(
                lambda x: enc.transform([x])[0] if x in known_classes else 0
            )
    return df

def transform_input_dataframe(df_raw: pd.DataFrame, encoders: dict, feature_names: list) -> pd.DataFrame:
    """End-to-end preprocessing pipeline from raw customer DataFrame to model feature matrix."""
    df = df_raw.copy()
    df = clean_total_charges(df)
    df = engineer_features(df)
    df = encode_features(df, encoders)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]
