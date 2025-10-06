"""Allocate grams across ingredients based on reweighted confidences and area.

- No hard-coded labels; operates on provided detections.
- Uses mask polygon area if available, else bbox area, else 1.
"""
from __future__ import annotations

from typing import Dict, List, Mapping


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    # Shoelace formula
    if not poly or len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _box_area(box: list | tuple) -> float:
    if not box or len(box) != 4:
        return 0.0
    left, top, right, bottom = box
    try:
        w = max(0.0, float(right) - float(left))
        h = max(0.0, float(bottom) - float(top))
        return w * h
    except Exception:
        return 0.0


def allocate_grams_per_ingredient(
    detections: List[dict],
    total_grams: float | None,
    blend_lambda: float = 0.3,
    beta: float = 0.5,
    gamma: float = 1.0,
    min_grams: float = 0.0,
    rounding: str = "nearest_gram",
) -> Dict[str, float]:
    """Return per-ingredient grams using area/confidence blended weights.

    If total_grams is None, compute weights but return 0 for all.
    """
    # Aggregate per ingredient
    per_ing_area: Dict[str, float] = {}
    per_ing_conf: Dict[str, float] = {}

    for det in detections:
        lbl = str(det.get("label", "")).strip().lower()
        try:
            conf = float(det.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        area = 0.0
        mask_poly = det.get("mask_polygon")
        box = det.get("box")
        if isinstance(mask_poly, list) and len(mask_poly) >= 3:
            try:
                area = _polygon_area([(float(x), float(y)) for x, y in mask_poly])
            except Exception:
                area = 0.0
        if area <= 0 and isinstance(box, (list, tuple)):
            area = _box_area(box)
        if area <= 0:
            area = 1.0
        per_ing_area[lbl] = per_ing_area.get(lbl, 0.0) + float(area)
        per_ing_conf[lbl] = per_ing_conf.get(lbl, 0.0) + float(conf)

    # Compute blended weights
    weights: Dict[str, float] = {}
    for lbl in set(per_ing_area.keys()) | set(per_ing_conf.keys()):
        A = max(1e-9, per_ing_area.get(lbl, 0.0)) ** beta
        C = max(1e-9, per_ing_conf.get(lbl, 0.0)) ** gamma
        # Blend between area and confidence
        w = (A ** blend_lambda) * (C ** (1.0 - blend_lambda))
        weights[lbl] = max(0.0, float(w))

    total_w = sum(weights.values())
    if total_w <= 0:
        return {lbl: 0.0 for lbl in weights.keys()}

    grams: Dict[str, float] = {}
    for lbl, w in weights.items():
        g = (float(total_grams) * (w / total_w)) if (total_grams is not None) else 0.0
        if g < min_grams:
            g = 0.0 if total_grams is None else min_grams
        if rounding == "nearest_gram":
            g = float(round(g))
        grams[lbl] = float(g)

    return grams
