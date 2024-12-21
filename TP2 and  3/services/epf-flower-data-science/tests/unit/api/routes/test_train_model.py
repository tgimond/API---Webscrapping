import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open
from src.api.routes.data import router
from src.services.data import load_iris_dataset, process_iris_dataset, split_iris_dataset, train_model
from src.app import get_application
import pandas as pd
from io import StringIO

app = get_application()
app.include_router(router, prefix="/data")

client = TestClient(app)

@pytest.fixture
def mock_dataset_file():
    dataset_data = """sepal_length,sepal_width,petal_length,petal_width,Species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
"""
    with patch("builtins.open", mock_open(read_data=dataset_data)):
        yield

def test_train_model_success(mock_dataset_file):
    df = load_iris_dataset()
    df_processed = process_iris_dataset(df)
    split_data = split_iris_dataset(df_processed)
    X_train = pd.read_json(StringIO(split_data["X_train"]), orient='split')
    y_train = pd.read_json(StringIO(split_data["y_train"]), orient='split').squeeze()
    model_path = train_model(X_train, y_train)
    assert os.path.exists(model_path)

def test_train_iris_model_endpoint(mock_dataset_file):
    response = client.post("/data/train-iris-model")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model_path" in data
    assert os.path.exists(data["model_path"])