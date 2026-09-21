import pytest
from fastapi.testclient import TestClient
import os
import sys

# Set mock or import app from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

@pytest.fixture(scope="module")
def client():
    """Fixture to provide a TestClient with startup/shutdown lifespan events executed."""
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    """Test that the GET / root endpoint returns a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "Welcome" in json_data["message"]
    assert json_data["status"] == "Running"

def test_predict_churn_valid_low_risk(client):
    """Test POST /predict with a valid profile expected to be low risk."""
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "churn_probability" in json_data
    assert "risk_level" in json_data
    assert "recommendation" in json_data
    assert json_data["risk_level"] in ["Low", "Medium", "High"]

def test_predict_batch_valid(client):
    """Test POST /predict/batch with a list of customer profiles."""
    payload = [
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.0,
            "TotalCharges": 2700.0
        },
        {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 190.0
        }
    ]
    
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert data["high_risk_count"] + data["medium_risk_count"] + data["low_risk_count"] == 2

def test_predict_churn_invalid_category(client):
    """Test POST /predict with an invalid category value, verifying 400 Bad Request."""
    payload = {
        "gender": "Non-Binary",  # Invalid gender (supported: Female, Male)
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid value" in response.json()["detail"]

def test_predict_churn_missing_fields(client):
    """Test POST /predict with missing fields, verifying 422 Unprocessable Entity (Pydantic validation)."""
    payload = {
        "gender": "Male",
        "tenure": 12
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
