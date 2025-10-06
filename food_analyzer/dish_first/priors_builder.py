"""Utilities to derive dish labels and dish→ingredient priors from ground truth.

No hard-coded dishes or ingredients. Reads ground_truth.json and optional
ingredient label set to build:
- dish_labels.json (list of dish labels = plate types)
- dish_to_ingredient.json (mapping dish -> ingredient weights)

CLI usage:
  python -m food_analyzer.dish_first.priors_builder \
    --ground-truth ground_truth.json \
    --ingredient-labels ingredient_labels.json \
    --out-priors priors/dish_to_ingredient.json \
    --out-dishes dish_labels.json \
    --normalize l1 \
    --smoothing-eps 0.001

Notes:
- ingredient_labels.json should be a JSON list of strings (the classifier label space).
- If not provided, priors will be built over the union of all ingredients in ground truth and left as-is (no normalization to your classifier space).
- For best results, provide ingredient labels so we can map via LabelNormalizer and drop unmatched terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from food_analyzer.utils.labels import LabelNormalizer, load_synonym_map


def _read_json_list(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return [str(x).strip() for x in data if str(x).strip()]


def _load_ground_truth(path: Path) -> Mapping[str, Iterable[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ground_truth.json must map plate_type -> [ingredients]")
    # Coerce values to list of strings
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
    return out


def derive_dishes_and_priors(
    ground_truth_path: Path,
    ingredient_labels_path: Optional[Path] = None,
    synonyms_path: Optional[Path] = None,
    smoothing_eps: float = 1e-3,
    normalize: str = "l1",  # l1|softmax
    min_count: int = 0,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Return (dish_labels, priors_table) derived from ground truth.

    If ingredient_labels_path is provided, ingredients are normalized/mapped to
    that space using LabelNormalizer and unknowns are dropped. Otherwise, the
    ingredient space is the set union present in ground truth.
    """
    gt = _load_ground_truth(ground_truth_path)
    dish_labels = sorted(list(gt.keys()))

    # Ingredient space and normalizer setup
    ingredient_space: List[str]
    normalizer: Optional[LabelNormalizer] = None

    if ingredient_labels_path is not None:
        ing_labels = _read_json_list(ingredient_labels_path)
        syn = load_synonym_map(synonyms_path or Path("dynamic_synonyms.json"))
        normalizer = LabelNormalizer.from_labels(ing_labels, syn)
        ingredient_space = ing_labels
    else:
        # Use union of ingredients in GT; no normalization
        union: set[str] = set()
        for items in gt.values():
            for x in items:
                union.add(str(x).strip().lower())
        ingredient_space = sorted(list(union))

    # Count and weight per dish
    table: Dict[str, Dict[str, float]] = {}
    for dish in dish_labels:
        raw_items = gt.get(dish, [])
        counts: Dict[str, int] = {}
        for x in raw_items:
            if normalizer is not None:
                canon = normalizer.normalize(x)
                if not canon:
                    continue
                key = canon
            else:
                key = str(x).strip().lower()
            counts[key] = counts.get(key, 0) + 1
        # Apply min_count filter
        if min_count > 0:
            counts = {k: v for k, v in counts.items() if v >= min_count}
        # Smooth and normalize
        weights: Dict[str, float] = {}
        if normalize == "softmax":
            import math
            if counts:
                m = max(counts.values())
            else:
                m = 0
            exps = {k: math.exp((counts.get(k, 0) - m)) for k in ingredient_space}
            total = sum(exps.values())
            weights = {k: (exps[k] / total if total > 0 else 0.0) for k in ingredient_space}
        else:
            # l1 with smoothing
            total = sum(counts.values()) + smoothing_eps * max(1, len(ingredient_space))
            for k in ingredient_space:
                weights[k] = ((counts.get(k, 0) + smoothing_eps) / total) if total > 0 else 0.0
        table[dish] = weights

    return dish_labels, table


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build dish labels and priors from ground truth")
    p.add_argument("--ground-truth", required=True, help="Path to ground_truth.json")
    p.add_argument("--ingredient-labels", default=None, help="Path to ingredient_labels.json (list)")
    p.add_argument("--synonyms", default=None, help="Optional synonyms JSON mapping for normalization")
    p.add_argument("--out-priors", required=True, help="Path to write dish_to_ingredient.json")
    p.add_argument("--out-dishes", required=True, help="Path to write dish_labels.json")
    p.add_argument("--normalize", choices=["l1", "softmax"], default="l1")
    p.add_argument("--smoothing-eps", type=float, default=1e-3)
    p.add_argument("--min-count", type=int, default=0)
    args = p.parse_args(argv)

    gt_path = Path(args.ground_truth)
    ing_path = Path(args.ingredient_labels) if args.ingredient_labels else None
    syn_path = Path(args.synonyms) if args.synonyms else None
    out_priors = Path(args.out_priors)
    out_dishes = Path(args.out_dishes)

    dishes, priors = derive_dishes_and_priors(
        ground_truth_path=gt_path,
        ingredient_labels_path=ing_path,
        synonyms_path=syn_path,
        smoothing_eps=float(args.smoothing_eps),
        normalize=str(args.normalize),
        min_count=int(args.min_count),
    )

    _write_json(out_dishes, dishes)
    _write_json(out_priors, priors)
    print(f"Wrote: {out_dishes} and {out_priors}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
