import os
from contextlib import asynccontextmanager
from typing import List
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')

# Global variables for models and resources
model = None
encoders = None
feature_names = None

def init_resources():
    """Load model, encoders, and feature names."""
    global model, encoders, feature_names
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(
            "Required model resources are missing. Please run data_prep.py and train.py first."
        )
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("Model resources successfully loaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown resource management."""
    try:
        init_resources()
    except Exception as e:
        print(f"Error loading resources on startup: {str(e)}")
        raise RuntimeError(f"Server startup failed due to missing or corrupted model resources: {str(e)}")
    yield

# Initialize FastAPI application with lifespan
app = FastAPI(
    title="Customer Churn Prediction API",
    description="A REST API for predicting customer churn risk using machine learning models.",
    version="1.1.0",
    lifespan=lifespan
)

# Define input schema using Pydantic (using json_schema_extra for Pydantic V2 compatibility)
class CustomerData(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "Female"}, description="Gender: 'Female', 'Male'")
    SeniorCitizen: int = Field(..., json_schema_extra={"example": 0}, description="SeniorCitizen: 0 (No), 1 (Yes)")
    Partner: str = Field(..., json_schema_extra={"example": "Yes"}, description="Partner: 'Yes', 'No'")
    Dependents: str = Field(..., json_schema_extra={"example": "No"}, description="Dependents: 'Yes', 'No'")
    tenure: int = Field(..., json_schema_extra={"example": 12}, description="Number of months the customer has stayed")
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"}, description="PhoneService: 'Yes', 'No'")
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"}, description="MultipleLines: 'Yes', 'No', 'No phone service'")
    InternetService: str = Field(..., json_schema_extra={"example": "Fiber optic"}, description="InternetService: 'DSL', 'Fiber optic', 'No'")
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"}, description="OnlineSecurity: 'Yes', 'No', 'No internet service'")
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"}, description="OnlineBackup: 'Yes', 'No', 'No internet service'")
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"}, description="DeviceProtection: 'Yes', 'No', 'No internet service'")
    TechSupport: str = Field(..., json_schema_extra={"example": "No"}, description="TechSupport: 'Yes', 'No', 'No internet service'")
    StreamingTV: str = Field(..., json_schema_extra={"example": "Yes"}, description="StreamingTV: 'Yes', 'No', 'No internet service'")
    StreamingMovies: str = Field(..., json_schema_extra={"example": "No"}, description="StreamingMovies: 'Yes', 'No', 'No internet service'")
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"}, description="Contract: 'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"}, description="PaperlessBilling: 'Yes', 'No'")
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"}, description="PaymentMethod: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'")
    MonthlyCharges: float = Field(..., json_schema_extra={"example": 70.35}, description="Monthly charges amount")
    TotalCharges: float = Field(..., json_schema_extra={"example": 844.20}, description="Total charges amount")

# Define single output schema
class ChurnPredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Churn risk probability represented as a fraction (0.00 to 1.00)")
    risk_level: str = Field(..., description="Risk tier: 'Low', 'Medium', 'High'")
    recommendation: str = Field(..., description="Actionable retention recommendations based on risk level")

# Define batch output schema
class BatchPredictionResponse(BaseModel):
    total_customers: int = Field(..., description="Total number of evaluated customers")
    high_risk_count: int = Field(..., description="Number of high churn risk customers")
    medium_risk_count: int = Field(..., description="Number of medium churn risk customers")
    low_risk_count: int = Field(..., description="Number of low churn risk customers")
    predictions: List[ChurnPredictionResponse] = Field(..., description="List of prediction results")

def transform_customer_to_features(customer: CustomerData, encoders_dict: dict, cols: list) -> pd.DataFrame:
    """Preprocesses and encodes customer profile data for model inference."""
    # 1. Feature Engineering
    services = [
        customer.PhoneService, customer.MultipleLines, customer.OnlineSecurity,
        customer.OnlineBackup, customer.DeviceProtection, customer.TechSupport,
        customer.StreamingTV, customer.StreamingMovies
    ]
    num_services = sum(1 for s in services if s == 'Yes')
    avg_monthly_spend = customer.MonthlyCharges if customer.tenure == 0 else customer.TotalCharges / customer.tenure
    is_new_customer = 1 if customer.tenure <= 6 else 0

    # Helper function to encode categorical string values safely using fitted LabelEncoders
    def encode_col(col_name: str, val: str) -> int:
        if col_name in encoders_dict:
            try:
                return int(encoders_dict[col_name].transform([str(val)])[0])
            except ValueError:
                allowed = list(encoders_dict[col_name].classes_)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value '{val}' for category '{col_name}'. Supported options: {allowed}"
                )
        return val

    encoded_features = {
        'gender': encode_col('gender', customer.gender),
        'SeniorCitizen': customer.SeniorCitizen,
        'Partner': encode_col('Partner', customer.Partner),
        'Dependents': encode_col('Dependents', customer.Dependents),
        'tenure': customer.tenure,
        'PhoneService': encode_col('PhoneService', customer.PhoneService),
        'MultipleLines': encode_col('MultipleLines', customer.MultipleLines),
        'InternetService': encode_col('InternetService', customer.InternetService),
        'OnlineSecurity': encode_col('OnlineSecurity', customer.OnlineSecurity),
        'OnlineBackup': encode_col('OnlineBackup', customer.OnlineBackup),
        'DeviceProtection': encode_col('DeviceProtection', customer.DeviceProtection),
        'TechSupport': encode_col('TechSupport', customer.TechSupport),
        'StreamingTV': encode_col('StreamingTV', customer.StreamingTV),
        'StreamingMovies': encode_col('StreamingMovies', customer.StreamingMovies),
        'Contract': encode_col('Contract', customer.Contract),
        'PaperlessBilling': encode_col('PaperlessBilling', customer.PaperlessBilling),
        'PaymentMethod': encode_col('PaymentMethod', customer.PaymentMethod),
        'MonthlyCharges': customer.MonthlyCharges,
        'TotalCharges': customer.TotalCharges,
        'NumServices': num_services,
        'AvgMonthlySpend': avg_monthly_spend,
        'IsNewCustomer': is_new_customer
    }
    return pd.DataFrame([encoded_features])[cols]

def evaluate_risk(churn_proba: float, contract_type: str) -> tuple[str, str]:
    """Determines risk level and actionable recommendation from probability."""
    churn_proba_pct = churn_proba * 100
    if churn_proba_pct < 30.0:
        risk_level = "Low"
        recommendation = "Low churn risk. Maintain standard service level and monitor periodically."
    elif churn_proba_pct < 60.0:
        risk_level = "Medium"
        recommendation = (
            "Medium churn risk. Recommend offering personalized engagement campaign, loyalty benefits, "
            "or reviewing general customer feedback."
        )
    else:
        risk_level = "High"
        recommendation = (
            f"High churn risk! Prompt action recommended. Consider: "
            f"1. Upgrading current '{contract_type}' contract to a 1 or 2-year term with a promotional discount. "
            f"2. Offering complementary digital protection/technical support services. "
            f"3. Initiating a proactive retention call."
        )
    return risk_level, recommendation

@app.get("/")
def read_root():
    """Welcome endpoint displaying basic API instructions."""
    return {
        "message": "Welcome to the Customer Churn Prediction API!",
        "instructions": "Send a POST request to '/predict' or '/predict/batch' with customer JSON data to evaluate churn risk.",
        "docs_url": "/docs",
        "status": "Running",
        "version": "1.1.0"
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerData):
    """Predict churn risk and level for a single customer profile."""
    global model, encoders, feature_names
    if model is None or encoders is None or feature_names is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not initialized. Please ensure resource files are present and restart the server."
        )
    try:
        df_input = transform_customer_to_features(customer, encoders, feature_names)
        churn_proba = float(model.predict_proba(df_input)[0][1])
        risk_level, recommendation = evaluate_risk(churn_proba, customer.Contract)
        return ChurnPredictionResponse(
            churn_probability=round(churn_proba, 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during risk processing: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(customers: List[CustomerData]):
    """Predict churn risk and level for a batch list of customer profiles."""
    global model, encoders, feature_names
    if model is None or encoders is None or feature_names is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not initialized. Please ensure resource files are present and restart the server."
        )
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list cannot be empty.")

    try:
        dfs = []
        for customer in customers:
            dfs.append(transform_customer_to_features(customer, encoders, feature_names))
        
        batch_df = pd.concat(dfs, ignore_index=True)
        probas = model.predict_proba(batch_df)[:, 1]

        predictions = []
        high_cnt, med_cnt, low_cnt = 0, 0, 0
        for i, customer in enumerate(customers):
            churn_proba = float(probas[i])
            risk_level, recommendation = evaluate_risk(churn_proba, customer.Contract)
            if risk_level == "High":
                high_cnt += 1
            elif risk_level == "Medium":
                med_cnt += 1
            else:
                low_cnt += 1

            predictions.append(ChurnPredictionResponse(
                churn_probability=round(churn_proba, 4),
                risk_level=risk_level,
                recommendation=recommendation
            ))

        return BatchPredictionResponse(
            total_customers=len(customers),
            high_risk_count=high_cnt,
            medium_risk_count=med_cnt,
            low_risk_count=low_cnt,
            predictions=predictions
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during batch risk processing: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
