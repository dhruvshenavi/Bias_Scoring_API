from statistics import median, mode, StatisticsError
from typing import List, Optional

def aggregate(scores: List[float]) -> dict:
    if not scores:
        return {
            "aggregate_score": 0.0,
            "aggregate_label": "neutral",
            "median_score": 0.0,
            "mode_value": None
        }

    avg = sum(scores) / len(scores)
    med = float(median(scores))

    # mode of sign-bucket labels is usually more stable than mode(score float)
    # but your schema has mode_value: str, so we can return label-mode
    labels = [score_to_label(s) for s in scores]
    try:
        mode_label = mode(labels)
    except StatisticsError:
        mode_label = None

    return {
        "aggregate_score": round(float(avg), 2),
        "aggregate_label": score_to_label(avg),
        "median_score": round(med, 2),
        "mode_value": mode_label
    }

def score_to_label(score: float) -> str:
    # Simple bands (tune as you like)
    if score >= 2.0:
        return "pro-BJP"
    if score <= -2.0:
        return "pro-Congress"
    return "neutral"