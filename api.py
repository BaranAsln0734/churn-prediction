import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="A REST API for predicting customer churn risk using machine learning models.",
    version="1.0.0"
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')

# Global variables for models and resources
model = None
encoders = None
feature_names = None

@app.on_event("startup")
def load_resources():
    """Load model, encoders, and feature names on startup."""
    global model, encoders, feature_names
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
            raise FileNotFoundError(
                "Required model resources are missing. Please run data_prep.py and train.py first."
            )
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        print("Model resources successfully loaded.")
    except Exception as e:
        print(f"Error loading resources on startup: {str(e)}")
        # Raise exception to halt server startup if resources cannot be loaded
        raise RuntimeError(f"Server startup failed due to missing or corrupted model resources: {str(e)}")

# Define input schema using Pydantic
class CustomerData(BaseModel):
    gender: str = Field(..., example="Female", description="Gender: 'Female', 'Male'")
    SeniorCitizen: int = Field(..., example=0, description="SeniorCitizen: 0 (No), 1 (Yes)")
    Partner: str = Field(..., example="Yes", description="Partner: 'Yes', 'No'")
    Dependents: str = Field(..., example="No", description="Dependents: 'Yes', 'No'")
    tenure: int = Field(..., example=12, description="Number of months the customer has stayed")
    PhoneService: str = Field(..., example="Yes", description="PhoneService: 'Yes', 'No'")
    MultipleLines: str = Field(..., example="No", description="MultipleLines: 'Yes', 'No', 'No phone service'")
    InternetService: str = Field(..., example="Fiber optic", description="InternetService: 'DSL', 'Fiber optic', 'No'")
    OnlineSecurity: str = Field(..., example="No", description="OnlineSecurity: 'Yes', 'No', 'No internet service'")
    OnlineBackup: str = Field(..., example="Yes", description="OnlineBackup: 'Yes', 'No', 'No internet service'")
    DeviceProtection: str = Field(..., example="No", description="DeviceProtection: 'Yes', 'No', 'No internet service'")
    TechSupport: str = Field(..., example="No", description="TechSupport: 'Yes', 'No', 'No internet service'")
    StreamingTV: str = Field(..., example="Yes", description="StreamingTV: 'Yes', 'No', 'No internet service'")
    StreamingMovies: str = Field(..., example="No", description="StreamingMovies: 'Yes', 'No', 'No internet service'")
    Contract: str = Field(..., example="Month-to-month", description="Contract: 'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(..., example="Yes", description="PaperlessBilling: 'Yes', 'No'")
    PaymentMethod: str = Field(..., example="Electronic check", description="PaymentMethod: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'")
    MonthlyCharges: float = Field(..., example=70.35, description="Monthly charges amount")
    TotalCharges: float = Field(..., example=844.20, description="Total charges amount")

# Define output schema
class ChurnPredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Churn risk probability represented as a fraction (0.00 to 1.00)")
    risk_level: str = Field(..., description="Risk tier: 'Low', 'Medium', 'High'")
    recommendation: str = Field(..., description="Actionable retention recommendations based on risk level")

@app.get("/")
def read_root():
    """Welcome endpoint displaying basic API instructions."""
    return {
        "message": "Welcome to the Customer Churn Prediction API!",
        "instructions": "Send a POST request to '/predict' with customer JSON data to evaluate churn risk.",
        "docs_url": "/docs",
        "status": "Running"
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerData):
    """Predict churn risk and level for a given customer profile."""
    global model, encoders, feature_names
    
    # Verify that model and resources are loaded
    if model is None or encoders is None or feature_names is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not initialized. Please ensure resource files are present and restart the server."
        )
        
    try:
        # 1. Feature Engineering
        # Calculate NumServices (Count 'Yes' across specific service columns)
        services = [
            customer.PhoneService, customer.MultipleLines, customer.OnlineSecurity,
            customer.OnlineBackup, customer.DeviceProtection, customer.TechSupport,
            customer.StreamingTV, customer.StreamingMovies
        ]
        num_services = sum(1 for s in services if s == 'Yes')
        
        # Calculate AvgMonthlySpend (Avoid division by zero if tenure is 0)
        avg_monthly_spend = customer.MonthlyCharges if customer.tenure == 0 else customer.TotalCharges / customer.tenure
        
        # Calculate IsNewCustomer (1 if tenure <= 6 months else 0)
        is_new_customer = 1 if customer.tenure <= 6 else 0
        
        # Helper function to encode categorical string values safely using fitted LabelEncoders
        def encode_categorical_field(column_name: str, value: str) -> int:
            if column_name in encoders:
                try:
                    return int(encoders[column_name].transform([str(value)])[0])
                except ValueError:
                    # Raise a 400 Bad Request for categories not matching original training categories
                    allowed_categories = list(encoders[column_name].classes_)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value '{value}' for category '{column_name}'. Supported options: {allowed_categories}"
                    )
            return value

        # 2. Map and encode the raw request inputs
        encoded_features = {
            'gender': encode_categorical_field('gender', customer.gender),
            'SeniorCitizen': customer.SeniorCitizen,
            'Partner': encode_categorical_field('Partner', customer.Partner),
            'Dependents': encode_categorical_field('Dependents', customer.Dependents),
            'tenure': customer.tenure,
            'PhoneService': encode_categorical_field('PhoneService', customer.PhoneService),
            'MultipleLines': encode_categorical_field('MultipleLines', customer.MultipleLines),
            'InternetService': encode_categorical_field('InternetService', customer.InternetService),
            'OnlineSecurity': encode_categorical_field('OnlineSecurity', customer.OnlineSecurity),
            'OnlineBackup': encode_categorical_field('OnlineBackup', customer.OnlineBackup),
            'DeviceProtection': encode_categorical_field('DeviceProtection', customer.DeviceProtection),
            'TechSupport': encode_categorical_field('TechSupport', customer.TechSupport),
            'StreamingTV': encode_categorical_field('StreamingTV', customer.StreamingTV),
            'StreamingMovies': encode_categorical_field('StreamingMovies', customer.StreamingMovies),
            'Contract': encode_categorical_field('Contract', customer.Contract),
            'PaperlessBilling': encode_categorical_field('PaperlessBilling', customer.PaperlessBilling),
            'PaymentMethod': encode_categorical_field('PaymentMethod', customer.PaymentMethod),
            'MonthlyCharges': customer.MonthlyCharges,
            'TotalCharges': customer.TotalCharges,
            'NumServices': num_services,
            'AvgMonthlySpend': avg_monthly_spend,
            'IsNewCustomer': is_new_customer
        }
        
        # 3. Create DataFrame and align column sequence with the training feature names
        df_input = pd.DataFrame([encoded_features])[feature_names]
        
        # 4. Predict probability and determine risk level and recommendation
        churn_proba = float(model.predict_proba(df_input)[0][1])
        churn_proba_pct = churn_proba * 100
        
        # Determine risk level and custom action steps based on thresholds (0-30-60-100)
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
                f"1. Upgrading current '{customer.Contract}' contract to a 1 or 2-year term with a promotional discount. "
                f"2. Offering complementary digital protection/technical support services. "
                f"3. Initiating a proactive retention call."
            )
            
        return ChurnPredictionResponse(
            churn_probability=round(churn_proba, 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        # Return a 500 error for internal prediction and dataframe manipulation faults
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during risk processing: {str(e)}"
        )

# Executable entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
