# from fastapi import APIRouter, HTTPException
# from app.schemas.inference import InferenceRequest, InferenceResponse, ItemScored
# from app.services.bias_scorer import compute_bias_score
# from app.utils.aggregates import aggregate
# from app.utils.cleaner import cleaner



# router = APIRouter()

# @router.post("/inference", response_model=InferenceResponse)
# def run_inference(req: InferenceRequest):
#     if not req.ner_list:
#         raise HTTPException(status_code=400, detail="ner_list is required and cannot be empty")

#     scored_list = []
#     scores = []

#     for item in req.ner_list:
#         sent = (item.sent or "").strip()
#         cleaned_text = cleaner(sent)

#         if not sent:
#             # skip empty sentences gracefully
#             continue

#         s =compute_bias_score(cleaned_text,)
#         scores.append(s)

#         scored_list.append(
#             ItemScored(sent=sent, label=item.label, score=s)
#         )

#     agg = aggregate(scores)

#     return InferenceResponse(
#         aggregate_score=agg["aggregate_score"],
#         aggregate_label=agg["aggregate_label"],
#         scored_list=scored_list,
#         median_score=agg["median_score"],
#         mode_value=agg["mode_value"]
#     )

# from fastapi import APIRouter, HTTPException
# from app.schemas.inference import InferenceRequest, InferenceResponse, ItemScored
# from app.services.bias_scorer import compute_bias_score
# from app.utils.aggregates import aggregate
# from app.utils.cleaner import cleaner

# router = APIRouter()


# @router.post("/inference", response_model=InferenceResponse)
# def run_inference(req: InferenceRequest):

#     if not req.ner_list:
#         raise HTTPException(
#             status_code=400,
#             detail="ner_list is required and cannot be empty"
#         )

#     scored_list = []
#     scores = []

#     for item in req.ner_list:
#         sent = (item.sent or "").strip()
#         target = (item.label or "").strip().lower()

#         if not sent:
#             continue  # skip empty

#         if target not in ["bjp", "congress"]:
#             # skip or handle invalid target
#             continue

#         cleaned_text = cleaner(sent)

#         # ✅ PASS TARGET HERE
#         s = compute_bias_score(cleaned_text, target)

#         scores.append(s)

#         scored_list.append(
#             ItemScored(
#                 sent=sent,
#                 label=item.label,   # original label preserved
#                 score=s
#             )
#         )

#     # handle case where all items got skipped
#     if not scores:
#         return InferenceResponse(
#             aggregate_score=0.0,
#             aggregate_label="neutral",
#             scored_list=[],
#             median_score=0.0,
#             mode_value="neutral"
#         )

#     agg = aggregate(scores)

#     return InferenceResponse(
#         aggregate_score=agg["aggregate_score"],
#         aggregate_label=agg["aggregate_label"],
#         scored_list=scored_list,
#         median_score=agg["median_score"],
#         mode_value=agg["mode_value"]
#     )

from fastapi import APIRouter, HTTPException

from app.schemas.inference import InferenceRequest, InferenceResponse, ItemScored
from app.services.bias_scorer import compute_target_axis_score
from app.utils.aggregates import aggregate
from app.utils.cleaner import cleaner

router = APIRouter()


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


@router.post("/inference", response_model=InferenceResponse)
def run_inference(req: InferenceRequest):
    if not req.ner_list:
        raise HTTPException(
            status_code=400,
            detail="ner_list is required and cannot be empty"
        )

    scored_list = []
    all_scores = []

    # target-local collections
    bjp_scores = []
    congress_scores = []

    for item in req.ner_list:
        sent = (item.sent or "").strip()
        target = (item.label or "").strip().lower()

        if not sent:
            continue

        if target not in ["bjp", "congress"]:
            continue

        cleaned_text = cleaner(sent)

        result = compute_target_axis_score(cleaned_text, target)

        target_score = float(result.get("score", 0.0))
        stance_label = result.get("label", "neutral")

        all_scores.append(target_score)

        if target == "bjp":
            bjp_scores.append(target_score)
        else:
            congress_scores.append(target_score)

        scored_list.append(
            ItemScored(
                sent=sent,
                target=target,
                label=stance_label,
                score=target_score
            )
        )

    if not scored_list:
        return InferenceResponse(
            bjp_axis=0.0,
            congress_axis=0.0,
            scored_list=[],
            median_score=0.0,
            mode_value="neutral"
        )

    bjp_axis = _safe_mean(bjp_scores)
    congress_axis = _safe_mean(congress_scores)

    # reuse existing utility for median/mode only
    agg = aggregate([
    {
        "sent": item.sent,
        "target": item.target,
        "label": item.label,
        "score": item.score
    }
    for item in scored_list
])

    return InferenceResponse(
        bjp_axis=bjp_axis,
        congress_axis=congress_axis,
        scored_list=scored_list,
        median_score=agg["median_score"],
        mode_value=agg["mode_value"]
    )