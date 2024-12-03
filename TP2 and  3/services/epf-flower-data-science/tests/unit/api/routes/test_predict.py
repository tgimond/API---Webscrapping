import os
import sys
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ajouter le chemin du projet à sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.api.routes.data import router
from src.app import get_application

app = get_application()
app.include_router(router, prefix="/data")

client = TestClient(app)

@pytest.fixture
def mock_model_file():
    model = RandomForestClassifier(n_estimators=10)
    X_train = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]]
    y_train = ["setosa", "setosa", "setosa"]
    model.fit(X_train, y_train)
    model_dir = os.path.join(os.path.dirname(__file__), '..\..\..\..\src\models')
    model_path = os.path.join(model_dir, 'iris_model.pkl')
    joblib.dump(model, model_path)
    yield
    os.remove(model_path)

def test_predict_success(mock_model_file):
    prediction_data = {
        "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
    }
    response = client.post("/data/predict", json=prediction_data)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

def test_predict_model_not_found():
    prediction_data = {
        "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
    }
    response = client.post("/data/predict", json=prediction_data)
    assert response.status_code == 404
    assert response.json() == {"detail": "Model file not found"}