# import re
# from app.core.config import settings
# from app.services.model_loader import get_pipeline



# def compute_bias_score(text: str) -> float:
#     """
#     Returns a bias score in [-10, +10]:
#       +10 strongly pro-BJP
#        0 neutral
#       -10 strongly pro-Congress
#     """
#     text = re.sub(r"\s+", " ", str(text)).strip()
#     if not text:
#         return 0.0


#     zs = get_pipeline()

#     out = zs(
#         sequences=text,
#         candidate_labels=settings.labels,
#         multi_label=True,
#         hypothesis_template=settings.hypothesis_template
#     )

#     scores = dict(zip(out["labels"], out["scores"]))

#     pro_bjp = scores.get("pro-BJP", 0.0)
#     anti_cong = scores.get("anti-Congress", 0.0)
#     pro_cong = scores.get("pro-Congress", 0.0)
#     anti_bjp = scores.get("anti-BJP", 0.0)
#     neutral = scores.get("neutral", 0.0)

#     positive = pro_bjp + anti_cong
#     negative = pro_cong + anti_bjp
#     denom = (positive + negative) + 1e-9

#     direction = (positive - negative) / denom

#     confidence = 1.0 - neutral
#     confidence = max(0.0, min(1.0, confidence))

#     score = 10.0 * direction * confidence
#     score = max(-10.0, min(10.0, float(score)))

#     return round(score, 2)

import torch
from app.services.cleaner import clean_text
from app.services.model_loader import (
    get_stance_model_components,
    get_intensity_pipeline,
)

TARGET_DISPLAY = {
    "bjp": "BJP",
    "congress": "Congress",
}

HYPOTHESIS_TEMPLATES = {
    "pro": "This sentence expresses support for {target}.",
    "anti": "This sentence expresses opposition to {target}.",
    "neutral": "This sentence expresses no clear opinion about {target}.",
}


def _map_intensity_to_magnitude(label: str) -> float:
    label = str(label).strip().lower()

    if label == "strong":
        return 8.0
    elif label == "mild":
        return 4.0
    return 0.0


def _predict_stance_scores(text: str, target: str, max_length: int = 256) -> dict:
    tokenizer, model = get_stance_model_components()
    device = next(model.parameters()).device

    target = str(target).strip().lower()
    if target not in TARGET_DISPLAY:
        raise ValueError(f"Invalid target: {target}")

    target_name = TARGET_DISPLAY[target]
    candidate_stances = ["pro", "anti", "neutral"]

    hypotheses = [
        HYPOTHESIS_TEMPLATES[s].format(target=target_name)
        for s in candidate_stances
    ]

    inputs = tokenizer(
        [text] * len(hypotheses),
        hypotheses,
        padding=True,
        truncation="only_first",
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    entailment_scores = probs[:, 1].detach().cpu().numpy()

    return {
        stance: float(score)
        for stance, score in zip(candidate_stances, entailment_scores)
    }


def _predict_intensity_label(text: str) -> str:
    intensity_pipe = get_intensity_pipeline()
    out = intensity_pipe(text, top_k=1)

    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        return out[0].get("label", "mild")

    return "mild"


def compute_bias_score(text: str, target: str) -> float:
    """
    Returns bias score in [-10, +10]

    +10 => strongly pro-BJP
      0 => neutral
    -10 => strongly pro-Congress
    """
    text = clean_text(text)
    if not text:
        return 0.0

    target = clean_text(target).lower()
    if target not in TARGET_DISPLAY:
        return 0.0

    # Stage 1: stance via NLI
    stance_scores = _predict_stance_scores(text=text, target=target)

    pro_score = stance_scores["pro"]
    anti_score = stance_scores["anti"]
    neutral_score = stance_scores["neutral"]

    direction_raw = pro_score - anti_score
    confidence = max(0.0, min(1.0, 1.0 - neutral_score))

    target_relative_score = 10.0 * direction_raw * confidence
    target_relative_score = max(-10.0, min(10.0, float(target_relative_score)))

    # Convert to project axis
    # BJP target: pro-BJP positive, anti-BJP negative
    # Congress target: pro-Congress should become negative, anti-Congress positive
    political_direction_score = (
        target_relative_score if target == "bjp" else -target_relative_score
    )

    # If stance is basically neutral, return 0 directly
    if abs(political_direction_score) < 1e-6 or confidence < 0.20:
        return 0.0

    # Stage 2: intensity
    intensity_label = _predict_intensity_label(text)
    magnitude = _map_intensity_to_magnitude(intensity_label)

    # Keep only sign from stage 1, magnitude from stage 2
    final_score = magnitude if political_direction_score > 0 else -magnitude

    return round(final_score, 2)