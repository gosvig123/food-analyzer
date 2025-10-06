"""Refine dish posterior using observed ingredient detections and priors.

No hard-coded labels. Combines CLIP dish posterior with a likelihood term
from observed ingredients via dish→ingredient priors.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple
import math


def refine_dish_posterior(
    dish_topk: List[Mapping[str, float]],
    detections: List[Mapping[str, object]],
    priors: Mapping[str, Mapping[str, float]] | None,
    ingredient_space: List[str],
    alpha_like: float = 1.0,
    max_topk: int = 5,
) -> List[Dict[str, float]]:
    """Return a refined dish_topk by combining prior-based likelihood with initial posterior.

    dish_topk: list of {label, confidence} from dish classifier
    detections: list of detection dicts with label and confidence
    priors: optional table dish -> ingredient -> prior weight (already normalized)
    ingredient_space: list of classifier label strings (canonical)
    alpha_like: strength of the likelihood term
    """
    if not dish_topk:
        return dish_topk

    # Build simple ingredient weight summary from detections
    w_ing: Dict[str, float] = {}
    for det in detections:
        lbl = str(det.get("label", "")).strip().lower()
        try:
            conf = float(det.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if not lbl:
            continue
        w_ing[lbl] = w_ing.get(lbl, 0.0) + conf

    # Compute refined scores per dish
    scored: List[Tuple[str, float]] = []
    for entry in dish_topk:
        dlabel = str(entry.get("label", ""))
        qd = float(entry.get("confidence", 0.0))
        if not dlabel:
            continue
        # Likelihood term from priors: exp(sum_i w_i * log(pi(i|d) + eps)))
        like_log = 0.0
        if priors is not None and dlabel in priors:
            table = priors[dlabel]
            for ing, w in w_ing.items():
                pi = float(table.get(ing, 0.0))
                if pi > 0:
                    like_log += w * math.log(pi)
        # Combine posterior and likelihood
        score = math.log(max(qd, 1e-9)) + alpha_like * like_log
        scored.append((dlabel, score))

    # Softmax over scores
    if not scored:
        return dish_topk
    m = max(s for _, s in scored)
    exps = [(d, math.exp(s - m)) for d, s in scored]
    total = sum(v for _, v in exps)
    refined = [{"label": d, "confidence": v / total if total > 0 else 0.0} for d, v in exps]
    refined.sort(key=lambda x: x["confidence"], reverse=True)
    return refined[: max_topk]