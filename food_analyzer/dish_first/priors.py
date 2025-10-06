"""Dish→ingredient priors loader and mixture builder.

- No hard-coded dishes or ingredients; loads from file when provided.
- Supports JSON mapping or CSV matrix. Falls back to uniform when unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import math
import json
from pathlib import Path


@dataclass
class PriorsConfig:
    source: str = "file"  # file|uniform
    path: str | None = None
    smoothing_eps: float = 1e-3
    normalize: str = "l1"  # l1|softmax


def _l1_normalize(vec: Dict[str, float], eps: float) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in vec.values()) + eps * max(1, len(vec))
    return {k: (max(0.0, v) + eps) / total for k, v in vec.items()}


def _softmax_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    import math
    xs = list(vec.values())
    if not xs:
        return {}
    m = max(xs)
    exps = {k: math.exp(v - m) for k, v in vec.items()}
    total = sum(exps.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in exps.items()}


class DishIngredientPriors:
    def __init__(self, cfg: PriorsConfig, table: Mapping[str, Mapping[str, float]] | None):
        self.cfg = cfg
        self.table = {str(d).strip(): {str(i).strip().lower(): float(v) for i, v in inner.items()} for d, inner in (table or {}).items()}

    @classmethod
    def from_config(cls, cfg_dict: dict, ingredient_labels: List[str]) -> "DishIngredientPriors":
        cfg = PriorsConfig(
            source=str(cfg_dict.get("source", "file")),
            path=cfg_dict.get("path"),
            smoothing_eps=float(cfg_dict.get("smoothing_eps", 1e-3)),
            normalize=str(cfg_dict.get("normalize", "l1")),
        )
        table = None
        if cfg.source == "file" and isinstance(cfg.path, str):
            table = _load_priors_file(cfg.path)
        elif cfg.source == "uniform":
            table = {}
        else:
            table = None
        return cls(cfg, table)

    def prior_for_dish(self, dish_label: str, ingredient_space: List[str]) -> Dict[str, float]:
        dish_key = str(dish_label).strip()
        base: Dict[str, float] = {}
        if self.table and dish_key in self.table:
            base = {k: float(v) for k, v in self.table[dish_key].items()}
        else:
            # uniform over ingredient space if missing
            n = max(1, len(ingredient_space))
            return {lbl.lower(): 1.0 / n for lbl in ingredient_space}
        # Ensure all ingredients present
        for lbl in ingredient_space:
            base.setdefault(lbl.lower(), 0.0)
        # Normalize per config
        if self.cfg.normalize == "l1":
            return _l1_normalize(base, self.cfg.smoothing_eps)
        return _softmax_normalize(base)


def _load_priors_file(path: str) -> Mapping[str, Mapping[str, float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Priors file not found: {p}")
    if p.suffix.lower() in {".json"}:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON priors must be an object mapping dish->ingredient->weight")
        # Validate nested mapping
        out: Dict[str, Dict[str, float]] = {}
        for dish, inner in data.items():
            if not isinstance(inner, dict):
                continue
            out[str(dish)] = {str(k): float(v) for k, v in inner.items()}
        return out
    # CSV: header=ingredient labels, first column=dish
    import csv
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return {}
    header = rows[0][1:]
    out: Dict[str, Dict[str, float]] = {}
    for row in rows[1:]:
        if not row:
            continue
        dish = row[0]
        vals = row[1:]
        inner: Dict[str, float] = {}
        for i, col in enumerate(header):
            try:
                inner[str(col)] = float(vals[i]) if i < len(vals) else 0.0
            except Exception:
                inner[str(col)] = 0.0
        out[str(dish)] = inner
    return out


def build_mixture_prior(dish_topk: List[Mapping[str, float]], priors: DishIngredientPriors | None, ingredient_labels: List[str]) -> Dict[str, float]:
    """Mix per-dish priors by dish posterior to get a single ingredient prior.

    dish_topk: list of {label, confidence}
    ingredient_labels: the ingredient label space to project into (strings)
    """
    if not ingredient_labels:
        return {}
    # Start with zeros
    mix: Dict[str, float] = {lbl.lower(): 0.0 for lbl in ingredient_labels}
    total_q = 0.0
    for d in dish_topk:
        lbl = str(d.get("label", ""))
        q = float(d.get("confidence", 0.0))
        if not lbl or q <= 0:
            continue
        total_q += q
        prior_d = priors.prior_for_dish(lbl, ingredient_labels) if priors is not None else {l.lower(): 1.0 / max(1, len(ingredient_labels)) for l in ingredient_labels}
        for k, v in prior_d.items():
            mix[k] = mix.get(k, 0.0) + q * float(v)
    # Normalize mixture (l1)
    if total_q > 0:
        s = sum(mix.values())
        if s > 0:
            return {k: v / s for k, v in mix.items()}
    # Fallback uniform
    n = max(1, len(ingredient_labels))
    return {lbl.lower(): 1.0 / n for lbl in ingredient_labels}
