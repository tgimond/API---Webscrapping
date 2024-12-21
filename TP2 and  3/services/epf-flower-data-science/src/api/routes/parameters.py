from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))


from firestore import FirestoreClient

router = APIRouter()

class Parameters(BaseModel):
    """
    Pydantic model for parameters.
    
    Attributes:
        n_estimators (int): The number of trees in the forest.
        criterion (str): The function to measure the quality of a split.
    """
    n_estimators: int
    criterion: str

@router.post("/create-firestore-collection")
async def create_collection():
    """
    Create the Firestore collection 'parameters' with the specified document.

    Returns:
        dict: A message indicating the success of the collection creation.

    Raises:
        HTTPException: If there is an error creating the collection.
    """
    try:
        firestore_client = FirestoreClient()
        firestore_client.create_parameters_collection()
        return {"message": "Firestore collection created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-parameters")
async def get_parameters():
    """
    Get the parameters from the Firestore collection.

    Returns:
        dict: The parameters stored in the Firestore collection.

    Raises:
        HTTPException: If there is an error retrieving the parameters.
    """
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
    """
    Update the parameters document in the Firestore collection.

    Args:
        parameters (Parameters): The parameters to update.

    Returns:
        dict: A message indicating the success of the update.

    Raises:
        HTTPException: If there is an error updating the parameters.
    """
    try:
        firestore_client = FirestoreClient()
        firestore_client.update_parameters(parameters.dict())
        return {"message": "Parameters updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))