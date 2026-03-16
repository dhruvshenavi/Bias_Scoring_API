import torch
from transformers import pipeline
from app.core.config import settings

_zs_pipeline = None

def get_pipeline():
    """
    Returns a cached HF pipeline. Loads it once.
    """
    global _zs_pipeline
    if _zs_pipeline is not None:
        return _zs_pipeline

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else None

    _zs_pipeline = pipeline(
        "zero-shot-classification",
        model=settings.model_id,
        device=device,
        torch_dtype=torch_dtype
    )
    return _zs_pipeline