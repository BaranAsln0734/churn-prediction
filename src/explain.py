import os
import joblib
import pandas as pd
import numpy as np

# Using Agg backend to save plots without requiring a GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def explain_model():
    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Define file paths
    test_path = os.path.join(project_dir, 'data', 'processed', 'test.csv')
    x_test_path = os.path.join(project_dir, 'data', 'processed', 'X_test.csv')
    model_path = os.path.join(project_dir, 'models', 'best_model.pkl')
    feature_names_path = os.path.join(project_dir, 'models', 'feature_names.pkl')
    models_dir = os.path.join(project_dir, 'models')
    
    # 1. Read data/processed/X_test.csv (fall back to test.csv if not found)
    print("Loading test data...")
    if os.path.exists(x_test_path):
        X_test = pd.read_csv(x_test_path)
    elif os.path.exists(test_path):
        # Extract X_test by dropping the target column (Churn) if test.csv exists
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=['Churn']) if 'Churn' in test_df.columns else test_df
    else:
        raise FileNotFoundError(
            "Test data not found. Please make sure to run data preprocessing first."
        )
        
    # 2. Load trained model from models/best_model.pkl
    print("Loading trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model not found: {model_path}. Please train the model first."
        )
    model = joblib.load(model_path)
    
    # 3. Load models/feature_names.pkl (auto-create if missing)
    print("Loading feature names...")
    if not os.path.exists(feature_names_path):
        # If file is missing, extract feature names from X_test and save them
        feature_names = X_test.columns.tolist()
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(feature_names, feature_names_path)
        print(f"feature_names.pkl not found, dynamically created using X_test columns: {feature_names_path}")
    else:
        feature_names = joblib.load(feature_names_path)
        
    # Ensure X_test columns match feature_names
    X_test = X_test[feature_names]
    
    # 5. Calculate SHAP values on the first 200 rows of the test set (sampled for performance)
    sample_size = min(200, len(X_test))
    print(f"Sampling the first {sample_size} rows for SHAP calculation...")
    X_test_sample = X_test.iloc[:sample_size]
    
    # 4. Create SHAP explainer (Auto-detect model type)
    model_type = model.__class__.__name__
    print(f"Detected model type: {model_type}")
    
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        print("Using shap.TreeExplainer for tree-based model...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
    elif isinstance(model, LogisticRegression):
        print("Using shap.LinearExplainer for linear model...")
        explainer = shap.LinearExplainer(model, X_test_sample)
        shap_values = explainer.shap_values(X_test_sample)
    else:
        print("Using shap.Explainer for general model...")
        explainer = shap.Explainer(model, X_test_sample)
        shap_values = explainer(X_test_sample)
        
    # If SHAP output is an Explanation object, extract numpy array values
    if hasattr(shap_values, 'values'):
        shap_values = shap_values.values
        
    # Handle output dimension differences in multi-class/binary classification (especially for Random Forest)
    if isinstance(shap_values, list):
        # If list is returned, extract SHAP values for class 1 (Churn)
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # If 3D array is returned [samples, features, classes], select Class 1
        shap_values = shap_values[:, :, 1]
        
    # 6. Create and save two plots
    shap_feature_importance_path = os.path.join(models_dir, 'shap_feature_importance.png')
    shap_summary_path = os.path.join(models_dir, 'shap_summary.png')
    
    # a. SHAP summary plot (bar type) - Average impact
    print("Creating SHAP Feature Importance (Bar type) plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("Average Feature Impact on Churn Prediction (SHAP)", fontsize=14, pad=15)
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(shap_feature_importance_path, dpi=300)
    plt.close()
    print(f"SHAP feature importance plot saved: {shap_feature_importance_path}")
    
    # b. SHAP summary plot (beeswarm/dot type) - Detailed distribution
    print("Creating SHAP summary plot (beeswarm)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("Detailed Feature Value Impact on Churn Prediction", fontsize=14, pad=15)
    plt.xlabel("SHAP Value (impact on model output)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    
    # Update the colorbar label on the right to English
    fig = plt.gcf()
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            ax.set_ylabel("Feature Value (High -> Red, Low -> Blue)", rotation=270, labelpad=15, fontsize=10)
            
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=300)
    plt.close()
    print(f"SHAP beeswarm plot saved: {shap_summary_path}")
    
    # 7. Print the top 5 most influential features with names and mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    print("\n" + "="*50)
    print("   TOP 5 FEATURES INFLUENCING MODEL PREDICTIONS")
    print("="*50)
    for idx, (i, row) in enumerate(shap_ranking.head(5).iterrows(), 1):
        print(f"{idx}. {row['Feature']:<25}: {row['Mean_Abs_SHAP']:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    explain_model()
