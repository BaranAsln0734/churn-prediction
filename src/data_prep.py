import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df, categorical_cols, models_dir):
    """
    Encodes categorical columns using LabelEncoder and
    saves the fitted encoder objects to models/encoders.pkl file.
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded column: {col}")
        
    os.makedirs(models_dir, exist_ok=True)
    encoders_path = os.path.join(models_dir, 'encoders.pkl')
    joblib.dump(encoders, encoders_path)
    print(f"Saved encoder objects to: {encoders_path}")
    return df

def prepare_data():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    raw_data_path = os.path.join(project_dir, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    models_dir = os.path.join(project_dir, 'models')
    
    print(f"Reading raw data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Raw data file not found at {raw_data_path}. Please make sure to download the Telco Customer Churn dataset "
            "and place it in that directory."
        )
        
    df = pd.read_csv(raw_data_path)
    
    # 1. Clear whitespace in TotalCharges, convert to numeric
    print("Cleaning TotalCharges column and converting to numeric type...")
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Set TotalCharges to 0.0 for new customers with tenure = 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 2. Drop customerID column
    print("Dropping customerID column...")
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # 3. Convert Churn target to binary (0/1)
    print("Converting Churn target variable to binary type...")
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # 4. Engineer new features
    print("Engineering new features...")
    # NumServices: how many services the customer has (PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)
    
    # AvgMonthlySpend (MonthlyCharges if tenure is 0, else TotalCharges / tenure)
    df['AvgMonthlySpend'] = np.where(df['tenure'] == 0, df['MonthlyCharges'], df['TotalCharges'] / df['tenure'])
    
    # IsNewCustomer (tenure <= 6 months)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    
    # 5. Encode categorical columns using LabelEncoder
    print("Encoding categorical features...")
    # Find object columns to exclude target and already numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical columns and save encoders
    df = encode_categoricals(df, categorical_cols, models_dir)
        
    # 6. Split dataset into train and test sets (80/20, stratify=y)
    print("Splitting dataset into train and test sets...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Combine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # 7. Save results to data/processed/ folder as csv (also save X_test.csv and y_test.csv separately)
    os.makedirs(processed_dir, exist_ok=True)
    
    train_save_path = os.path.join(processed_dir, 'train.csv')
    test_save_path = os.path.join(processed_dir, 'test.csv')
    
    print(f"Saving train and test datasets to {processed_dir}...")
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    
    # Save X_test and y_test separately for compatibility with explain.py and business_impact.py
    X_test_path = os.path.join(processed_dir, 'X_test.csv')
    y_test_path = os.path.join(processed_dir, 'y_test.csv')
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print(f"X_test.csv and y_test.csv successfully created.")
    
    print("Data preprocessing steps complete!")

if __name__ == '__main__':
    prepare_data()
