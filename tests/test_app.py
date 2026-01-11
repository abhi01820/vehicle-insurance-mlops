import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_prediction_endpoint_structure():
    test_payload = {
        "Gender": "Male",
        "Age": "40",
        "Driving_License": "1",
        "Region_Code": "28.0",
        "Previously_Insured": "0",
        "Annual_Premium": "55555.0",
        "Policy_Sales_Channel": "26.0",
        "Vintage": "520",
        "Vehicle_Age": "> 2 Years",
        "Vehicle_Damage": "Yes"
    }
    
    response = client.post("/", data=test_payload)
    assert response.status_code == 200
