from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))


from firestore import FirestoreClient

router = APIRouter()

class Parameters(BaseModel):
    n_estimators: int
    criterion: str

@router.post("/create-firestore-collection")
async def create_collection():
    try:
        firestore_client = FirestoreClient()
        firestore_client.create_parameters_collection()
        return {"message": "Firestore collection created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-parameters")
async def get_parameters():
    try:
        firestore_client = FirestoreClient()
        parameters = firestore_client.get("parameters", "parameters")
        return parameters
    except FileExistsError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-parameters")
async def update_parameters(parameters: Parameters):
    try:
        firestore_client = FirestoreClient()
        firestore_client.update_parameters(parameters.dict())
        return {"message": "Parameters updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))