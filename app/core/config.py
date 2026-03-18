from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    #Hugging Face Model Repo
    model_id: str = "dhruvshenavi/distilbert-mnli-news-bias"

    #Optional (only if private repo)
    hf_token: Optional[str] = None

    #Inference settings
    hypothesis_template: str = "This text expresses a {} political bias."

    labels: List[str] = [
        "pro-BJP",
        "anti-BJP",
        "pro-Congress",
        "anti-Congress",
        "neutral"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()