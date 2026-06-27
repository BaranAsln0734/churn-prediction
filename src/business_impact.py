import os
import joblib
import pandas as pd
import numpy as np

def calculate_business_impact():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Define file paths
    test_path = os.path.join(project_dir, 'data', 'processed', 'test.csv')
    x_test_path = os.path.join(project_dir, 'data', 'processed', 'X_test.csv')
    y_test_path = os.path.join(project_dir, 'data', 'processed', 'y_test.csv')
    model_path = os.path.join(project_dir, 'models', 'best_model.pkl')
    
    # 1. Read test data (supports X_test/y_test split or combined test.csv)
    print("Loading datasets...")
    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        X_test = pd.read_csv(x_test_path)
        y_test_data = pd.read_csv(y_test_path)
        # Convert target variable to series
        y_test = y_test_data.iloc[:, 0] if isinstance(y_test_data, pd.DataFrame) else y_test_data
    elif os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=['Churn'])
        y_test = test_df['Churn']
    else:
        raise FileNotFoundError(
            "Test data not found. Please run data preprocessing (data_prep.py) first."
        )
        
    # 2. Load trained model
    print("Loading trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found: {model_path}. Please train models (train.py) first."
        )
    model = joblib.load(model_path)
    
    # 3. Make predictions on test set
    y_pred = model.predict(X_test)
    
    # 4. Number of churned customers correctly identified by model (True Positive)
    true_positives = int(np.sum((y_test == 1) & (y_pred == 1)))
    total_customers = len(y_test)
    actual_churns = int(np.sum(y_test == 1))
    
    # 5. Calculate average monthly charges from MonthlyCharges column
    if 'MonthlyCharges' in X_test.columns:
        avg_monthly_charges = float(X_test['MonthlyCharges'].mean())
    else:
        # Fallback default value if MonthlyCharges is not found
        avg_monthly_charges = 64.76  # Telco dataset approximate average
        print("Warning: MonthlyCharges column not found, using default average spend.")
        
    # 6. Define assumptions and calculate retained revenue
    RETENTION_SUCCESS_RATE = 0.25        # Probability of a customer staying after intervention (25%)
    AVG_REMAINING_LIFETIME_MONTHS = 12   # Estimated additional months customer would have stayed (months)
    
    # Number of saved customers
    estimated_saved_customers = true_positives * RETENTION_SUCCESS_RATE
    
    # Retained revenue = saved customers * average monthly charges * remaining lifetime months
    saved_revenue = estimated_saved_customers * avg_monthly_charges * AVG_REMAINING_LIFETIME_MONTHS
    
    # 7. Print results to console
    print("\n" + "="*50)
    print("   CUSTOMER CHURN MODEL BUSINESS VALUE REPORT")
    print("="*50)
    print(f"Total Customers in Test Set                  : {total_customers}")
    print(f"Actual Churned Customers                     : {actual_churns}")
    print(f"Correctly Identified Churns (TP)             : {true_positives}")
    print(f"Average Monthly Customer Spend               : {avg_monthly_charges:.2f} $")
    print("-"*50)
    print("Business Assumptions Used:")
    print(f"  * Campaign Convincing/Success Rate         : {RETENTION_SUCCESS_RATE * 100:.0f}%")
    print(f"  * Average Remaining Lifetime (Add. Months) : {AVG_REMAINING_LIFETIME_MONTHS} months")
    print("-"*50)
    print(f"Estimated Saved Customers                    : {estimated_saved_customers:.1f}")
    print(f"Estimated Saved Revenue (Annual)             : {saved_revenue:,.2f} $")
    print("="*50)
    print("Important Note: This calculation is based only on the test dataset (20%).")
    print("To see the total impact on your actual customer base, you need to scale")
    print("this result based on your total customer base size and churn rate.")
    print("="*50 + "\n")

if __name__ == '__main__':
    calculate_business_impact()
