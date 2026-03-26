import os
from pydantic import BaseModel

class Settings(BaseModel):
    #model_id: str = os.getenv("MODEL_ID", "facebook/bart-large-mnli")

    stance_model_id: str = "dhruvshenavi/Robert_V5_mnli_newsbias"
    intensity_model_id: str = "dhruvshenavi/Roberta_Intensity_V5"
    device: int = 0  # auto-set in loader
    hypothesis_template: str = "This text expresses a {} political bias."
    labels: list[str] = ["pro-BJP", "anti-BJP", "pro-Congress", "anti-Congress", "neutral"]
    hf_token: str = "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"
settings = Settings()