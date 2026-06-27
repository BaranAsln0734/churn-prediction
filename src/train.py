import os
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def train_models():
    # Define paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    train_path = os.path.join(project_dir, 'data', 'processed', 'train.csv')
    test_path = os.path.join(project_dir, 'data', 'processed', 'test.csv')
    models_dir = os.path.join(project_dir, 'models')
    
    # 1. Read the processed train/test datasets
    print("Reading processed datasets...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed data files not found. Please make sure to run 'src/data_prep.py' first."
        )
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Split features and target
    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # 2. Address class imbalance with SMOTE (apply only to the training set)
    print("Applying SMOTE to balance the training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {X_train.shape}, Churn distribution: {np.bincount(y_train)}")
    print(f"Resampled training shape: {X_train_resampled.shape}, Churn distribution: {np.bincount(y_train_resampled)}")
    
    # 3. Train and compare Logistic Regression, Random Forest, and XGBoost models
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': model_name,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc
        })
        print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC_AUC: {roc_auc:.4f}")
        
    # Convert results to DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # 5. Save comparison table as models/comparison.csv
    comparison_path = os.path.join(models_dir, 'comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison to: {comparison_path}")
    
    # 4. Find the best model (based on ROC_AUC)
    best_row = comparison_df.loc[comparison_df['ROC_AUC'].idxmax()]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model based on ROC_AUC: {best_model_name} with ROC_AUC of {best_row['ROC_AUC']:.4f}")
    
    # Save the best model as models/best_model.pkl
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"Saved the best model to: {best_model_path}")
    
    # Save feature names list
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    joblib.dump(X_train.columns.tolist(), feature_names_path)
    print(f"Saved feature names to: {feature_names_path}")

if __name__ == '__main__':
    train_models()
