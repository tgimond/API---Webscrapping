import os
import pandas as pd
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

def load_iris_dataset() -> pd.DataFrame:
    """
    Load the Iris dataset from a CSV file.

    Returns:
        pd.DataFrame: The loaded Iris dataset.

    Raises:
        HTTPException: If the dataset file is not found or if there is an error loading the dataset.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data/iris')
    dataset_path = os.path.join(data_dir, 'Iris.csv')

    # Vérifie si le fichier du dataset existe
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")

    # Charge le dataset
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

    return df

def process_iris_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the Iris dataset by performing necessary transformations.

    Args:
        df (pd.DataFrame): The Iris dataset to process.

    Returns:
        pd.DataFrame: The processed Iris dataset.
    """
    # Renommer les colonnes contenant "iris"
    try:
        df_processed = df.rename(columns=lambda x: x.replace('iris-', ''))
        # Enlever "iris-" dans les noms des espèces
        df_processed['Species'] = df_processed['Species'].str.replace('iris-', '', case=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

    return df_processed

def split_iris_dataset(df: pd.DataFrame) -> dict:
    """
    Split the Iris dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The Iris dataset to split.

    Returns:
        dict: A dictionary containing the training and testing sets.
    """
    try:
        X = df.drop(columns=['Species'])
        y = df['Species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

    return {
        "X_train": X_train.to_json(orient='split'),
        "X_test": X_test.to_json(orient='split'),
        "y_train": y_train.to_json(orient='split'),
        "y_test": y_test.to_json(orient='split')
    }

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """
    Train a RandomForest model on the Iris dataset.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.

    Returns:
        str: The path to the saved model file.

    Raises:
        HTTPException: If there is an error loading model parameters or saving the model.
    """
    # Charger les paramètres du modèle depuis le fichier JSON
    config_path = os.path.join(os.path.dirname(__file__), '../config/model_parameters.json')
    try:
        with open(config_path, 'r') as config_file:
            model_params = json.load(config_file)["RandomForestClassifier"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model parameters: {str(e)}")

    # Initialiser le modèle avec les paramètres chargés
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'iris_model.pkl')
    joblib.dump(model, model_path)

    return model_path

def predict_with_model(data: list) -> list:
    """
    Make predictions using the trained model.

    Args:
        data (list): The input data for making predictions.

    Returns:
        list: The predicted labels.

    Raises:
        HTTPException: If the model file is not found or if there is an error making predictions.
    """
    model_path = os.path.join(os.path.dirname(__file__), '../models/iris_model.pkl')

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    try:
        data_df = pd.DataFrame(data)
        predictions = model.predict(data_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

    return predictions.tolist()