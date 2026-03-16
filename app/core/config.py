import os
from pydantic import BaseModel

class Settings(BaseModel):
    model_id: str = os.getenv("MODEL_ID", "facebook/bart-large-mnli")
    device: int = 0  # auto-set in loader
    hypothesis_template: str = "This text expresses a {} political bias."
    labels: list[str] = ["pro-BJP", "anti-BJP", "pro-Congress", "anti-Congress", "neutral"]

settings = Settings()