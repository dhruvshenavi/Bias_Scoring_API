from fastapi import APIRouter, HTTPException
from app.schemas.inference import InferenceRequest, InferenceResponse, ItemScored
from app.services.bias_scorer import compute_bias_score
from app.utils.aggregates import aggregate
from app.utils.cleaner import cleaner



router = APIRouter()

@router.post("/inference", response_model=InferenceResponse)
def run_inference(req: InferenceRequest):
    if not req.ner_list:
        raise HTTPException(status_code=400, detail="ner_list is required and cannot be empty")

    scored_list = []
    scores = []

    for item in req.ner_list:
        sent = (item.sent or "").strip()
        cleaned_text = cleaner(sent)

        if not sent:
            # skip empty sentences gracefully
            continue

        s =compute_bias_score(cleaned_text)
        scores.append(s)

        scored_list.append(
            ItemScored(sent=sent, label=item.label, score=s)
        )

    agg = aggregate(scores)

    return InferenceResponse(
        aggregate_score=agg["aggregate_score"],
        aggregate_label=agg["aggregate_label"],
        scored_list=scored_list,
        median_score=agg["median_score"],
        mode_value=agg["mode_value"]
    )