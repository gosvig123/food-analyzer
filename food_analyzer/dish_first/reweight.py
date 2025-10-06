"""Confidence reweighting using dish→ingredient priors.

- Works with top-1 detection confidences by scaling them with prior(label).
- No hard-coded labels; prior map is provided externally.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Tuple


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def reweight_and_rerank_detections(
    detections: List[dict],
    prior_map: Mapping[str, float],
    cfg: Mapping[str, object] | None = None,
) -> List[dict]:
    """Reweight candidate lists per detection and rerank; optionally expand with secondary candidates.

    Returns a possibly expanded list of detection dicts.
    """
    if not detections:
        return detections
    method = str((cfg or {}).get("method", "boost"))
    alpha = float((cfg or {}).get("alpha", 0.8))
    temperature = float((cfg or {}).get("temperature", 1.0))
    tau = 1.0 / max(1e-6, temperature)
    min_secondary = float((cfg or {}).get("min_secondary_conf", 0.15))
    max_secondary = int((cfg or {}).get("max_secondary", 1))

    out: List[dict] = []

    for det in detections:
        base_label = str(det.get("label", "")).strip()
        base_conf = float(det.get("confidence", 0.0)) if det.get("confidence") is not None else 0.0
        det["confidence_base"] = base_conf

        candidates = det.get("candidates")
        # If no candidates, fall back to in-place reweight of the primary label
        if not isinstance(candidates, list) or not candidates:
            prior = float(prior_map.get(base_label.lower(), 0.0))
            if method == "multiply":
                new_conf = base_conf * (prior ** alpha if prior > 0 else 0.0)
            elif method == "prob_blend":
                new_conf = (base_conf ** tau) * (prior ** alpha if prior > 0 else 0.0)
            else:
                new_conf = base_conf * (1.0 + alpha * prior)
            det["confidence"] = _clamp01(float(new_conf))
            out.append(det)
            continue

        # Compute reweighted score for each candidate
        scored: List[Tuple[dict, float]] = []
        for cand in candidates:
            c_label = str(cand.get("label", "")).strip()
            c_conf = float(cand.get("confidence", 0.0))
            prior = float(prior_map.get(c_label.lower(), 0.0))
            if method == "multiply":
                score = c_conf * (prior ** alpha if prior > 0 else 0.0)
            elif method == "prob_blend":
                score = (c_conf ** tau) * (prior ** alpha if prior > 0 else 0.0)
            else:
                score = c_conf * (1.0 + alpha * prior)
            scored.append((cand, float(score)))

        # Sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            out.append(det)
            continue

        # Replace primary label with best candidate
        best_cand, best_score = scored[0]
        det["label"] = best_cand.get("label")
        det["confidence"] = _clamp01(best_score)
        out.append(det)

        # Optionally expand with next candidates
        added = 0
        for cand, sc in scored[1:]:
            if added >= max_secondary:
                break
            if sc < min_secondary:
                break
            # Clone detection dict (shallow copy) with secondary candidate
            clone = dict(det)
            clone["label"] = cand.get("label")
            clone["confidence"] = _clamp01(sc)
            # Do not carry forward original candidates to avoid explosion
            clone.pop("candidates", None)
            out.append(clone)
            added += 1

    return out


def reweight_detections_inplace(
    detections: List[dict],
    prior_map: Mapping[str, float],
    cfg: Mapping[str, object] | None = None,
) -> None:
    """Mutate detection dicts to include confidence_base and reweighted confidence.

    The classifier currently returns only a single label+confidence per detection.
    We therefore scale the confidence by a function of the prior(label).

    method: "boost" (default) | "prob_blend" | "multiply"
      - boost:     conf' = clamp01(conf * (1 + alpha * prior(label)))
                   (never punishes; only boosts based on prior)
      - prob_blend: conf' = clamp01(conf ^ tau * prior(label) ^ alpha)
      - multiply:   conf' = clamp01(conf * prior(label) ^ alpha)

    Args:
      detections: list of {label:str, confidence:float, ...}
      prior_map: mapping from lowercased ingredient label -> prior probability (sum to 1)
      cfg: reweighting configuration (alpha, temperature, method)
    """
    if not detections or not prior_map:
        return
    method = str((cfg or {}).get("method", "boost"))
    alpha = float((cfg or {}).get("alpha", 0.8))
    temperature = float((cfg or {}).get("temperature", 1.0))
    tau = 1.0 / max(1e-6, temperature)

    for det in detections:
        label = str(det.get("label", "")).strip().lower()
        try:
            conf = float(det.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        det["confidence_base"] = conf
        prior = float(prior_map.get(label, 0.0))
        if method == "multiply":
            new_conf = conf * (prior ** alpha if prior > 0 else 0.0)
        elif method == "prob_blend":
            new_conf = (conf ** tau) * (prior ** alpha if prior > 0 else 0.0)
        else:
            # boost (default)
            new_conf = conf * (1.0 + alpha * prior)
        det["confidence"] = _clamp01(float(new_conf))
