from statistics import median, mode, StatisticsError
from typing import List, Optional


def score_to_target_label(target: str, score: float) -> str:
    target = str(target or "").strip().lower()

    if abs(score) < 1e-6:
        return "No implied Stance"

    if target == "bjp":
        if score > 0:
            return "positive-stance-towards-BJP"
        return "critical-stance-towards-BJP"

    if target == "congress":
        if score > 0:
            return "positive-stance-towards-Congress"
        return "critical-stance-towards-Congress"

    return "No implied Stance"


def aggregate(scored_list: List[dict]) -> dict:
    """
    scored_list expects items like:
    {
        "sent": "...",
        "target": "bjp" | "congress",
        "label": "pro" | "anti" | "neutral",
        "score": float
    }
    """
    if not scored_list:
        return {
            "median_score": 0.0,
            "mode_value": None
        }

    scores = [float(item.get("score", 0.0)) for item in scored_list]
    med = float(median(scores))

    labels = [
        score_to_target_label(
            item.get("target", ""),
            float(item.get("score", 0.0))
        )
        for item in scored_list
    ]

    try:
        mode_label = mode(labels)
    except StatisticsError:
        mode_label = None

    return {
        "median_score": round(med, 2),
        "mode_value": mode_label
    }