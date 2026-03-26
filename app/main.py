from fastapi import FastAPI
from app.api.v1.router import api_router
from app.services.model_loader import get_intensity_pipeline,get_stance_model_components 

app = FastAPI(title="Bias Scoring API", version="1.0")

@app.on_event("startup")
def startup():
    # warm load model once at boot
    get_stance_model_components()
    get_intensity_pipeline()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(api_router, prefix="/api/v1")