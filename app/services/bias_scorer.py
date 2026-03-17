import re
from app.core.config import settings
from app.services.model_loader import get_pipeline



def compute_bias_score(text: str) -> float:
    """
    Returns a bias score in [-10, +10]:
      +10 strongly pro-BJP
       0 neutral
      -10 strongly pro-Congress
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return 0.0


    zs = get_pipeline()

    out = zs(
        sequences=text,
        candidate_labels=settings.labels,
        multi_label=True,
        hypothesis_template=settings.hypothesis_template
    )

    scores = dict(zip(out["labels"], out["scores"]))

    pro_bjp = scores.get("pro-BJP", 0.0)
    anti_cong = scores.get("anti-Congress", 0.0)
    pro_cong = scores.get("pro-Congress", 0.0)
    anti_bjp = scores.get("anti-BJP", 0.0)
    neutral = scores.get("neutral", 0.0)

    positive = pro_bjp + anti_cong
    negative = pro_cong + anti_bjp
    denom = (positive + negative) + 1e-9

    direction = (positive - negative) / denom

    confidence = 1.0 - neutral
    confidence = max(0.0, min(1.0, confidence))

    score = 10.0 * direction * confidence
    score = max(-10.0, min(10.0, float(score)))

    return round(score, 2)