"""API Router for Fast API."""
from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from src.api.routes import hello

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])

app = FastAPI()
app.include_router(router)

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")