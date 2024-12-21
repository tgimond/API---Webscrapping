import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open
from src.api.routes.parameters import router, Parameters
from src.app import get_application

app = get_application()
app.include_router(router, prefix="/parameters")

client = TestClient(app)

@pytest.fixture
def mock_firestore_client():
    with patch("firestore.FirestoreClient") as MockFirestoreClient:
        yield MockFirestoreClient

def test_create_firestore_collection(mock_firestore_client):
    response = client.post("/parameters/create-firestore-collection")
    assert response.status_code == 200
    assert response.json() == {"message": "Firestore collection created successfully"}

def test_get_parameters(mock_firestore_client):
    mock_firestore_client.return_value.get.return_value = {
        "n_estimators": 100,
        "criterion": "gini"
    }
    response = client.get("/parameters/get-parameters")
    assert response.status_code == 200
    data = response.json()
    assert "n_estimators" in data
    assert "criterion" in data

def test_update_parameters(mock_firestore_client):
    parameters = {
        "n_estimators": 200,
        "criterion": "entropy"
    }
    response = client.put("/parameters/update-parameters", json=parameters)
    assert response.status_code == 200
    assert response.json() == {"message": "Parameters updated successfully"}