# import torch
# from transformers import pipeline
# from app.core.config import settings

# _zs_pipeline = None

# def get_pipeline():
#     """
#     Returns a cached HF pipeline. Loads it once.
#     """
#     global _zs_pipeline
#     if _zs_pipeline is not None:
#         return _zs_pipeline

#     use_gpu = torch.cuda.is_available()

#     if use_gpu:
#         print(torch.cuda.get_device_name(0))

#     device = 0 if use_gpu else -1
#     torch_dtype = torch.float16 if use_gpu else None

#     print("Using GPU:", torch.cuda.is_available())
#     print("Device:", device)


#     _zs_pipeline = pipeline(
#         "zero-shot-classification",
#         model=settings.model_id,
#         device=device,
#         torch_dtype=torch_dtype
#     )
#     return _zs_pipeline

# app/services/model_loader.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from app.core.config import settings

_stance_tokenizer = None
_stance_model = None
_intensity_pipeline = None


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_pipeline_device():
    return 0 if torch.cuda.is_available() else -1


def get_stance_model_components():
    global _stance_tokenizer, _stance_model

    if _stance_tokenizer is not None and _stance_model is not None:
        return _stance_tokenizer, _stance_model

    device = _get_device()

    print("Loading stance NLI model...")
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    _stance_tokenizer = AutoTokenizer.from_pretrained(
        settings.stance_model_id,
        token=settings.hf_token
    )

    _stance_model = AutoModelForSequenceClassification.from_pretrained(
        settings.stance_model_id,
        token=settings.hf_token
    )

    _stance_model.to(device)
    _stance_model.eval()

    return _stance_tokenizer, _stance_model


def get_intensity_pipeline():
    global _intensity_pipeline

    if _intensity_pipeline is not None:
        return _intensity_pipeline

    device = _get_pipeline_device()

    print("Loading intensity model...")
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    _intensity_pipeline = pipeline(
        task="text-classification",
        model=settings.intensity_model_id,
        tokenizer=settings.intensity_model_id,
        device=device,
        token=settings.hf_token
    )

    return _intensity_pipeline