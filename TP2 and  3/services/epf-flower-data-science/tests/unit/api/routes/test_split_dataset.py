import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open
from src.api.routes.data import router
from src.services.data import load_iris_dataset, split_iris_dataset
from src.app import get_application

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

def test_split_iris_dataset_success(mock_dataset_file):
    df = load_iris_dataset()
    split_data = split_iris_dataset(df)
    assert "X_train" in split_data
    assert "X_test" in split_data
    assert "y_train" in split_data
    assert "y_test" in split_data

def test_split_iris_dataset_endpoint(mock_dataset_file):
    response = client.get("/data/split-iris-dataset")
    assert response.status_code == 200
    data = response.json()
    assert "X_train" in data
    assert "X_test" in data
    assert "y_train" in data
    assert "y_test" in data