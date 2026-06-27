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
    # If the model files are not trained yet, startup might fail or model is None, 
    # but we assume resources are loaded during testing.
    # In case the model file isn't present in testing environment:
    if response.status_code == 503:
        pytest.skip("Model files are not generated yet, skipping prediction test.")
        
    assert response.status_code == 200
    json_data = response.json()
    assert "churn_probability" in json_data
    assert "risk_level" in json_data
    assert "recommendation" in json_data
    assert json_data["risk_level"] in ["Low", "Medium", "High"]

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
    if response.status_code == 503:
        pytest.skip("Model files are not generated yet, skipping prediction test.")
        
    assert response.status_code == 400
    assert "Invalid value" in response.json()["detail"]

def test_predict_churn_missing_fields(client):
    """Test POST /predict with missing fields, verifying 422 Unprocessable Entity (Pydantic validation)."""
    payload = {
        "gender": "Male",
        "tenure": 12
        # Missing other fields
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
