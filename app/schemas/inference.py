from pydantic import BaseModel
from typing import List, Optional

class Item(BaseModel):
    sent: Optional[str] = None
    label: Optional[str] = None

class ItemScored(BaseModel):
    sent: Optional[str] = None
    label: Optional[str] = None
    score: Optional[float] = None

class InferenceRequest(BaseModel):
    # list of NER-targeted sentences (or entity-marked sentences)
    ner_list: Optional[List[Item]] = None

class InferenceResponse(BaseModel):
    aggregate_score: Optional[float] = None
    aggregate_label: Optional[str] = None
    scored_list: Optional[List[ItemScored]] = None
    median_score: Optional[float] = None
    mode_value: Optional[str] = None