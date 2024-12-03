import os
import pandas as pd
from fastapi import HTTPException

def load_iris_dataset() -> pd.DataFrame:
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