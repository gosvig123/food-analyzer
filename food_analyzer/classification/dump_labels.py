"""Dump the current classifier ingredient labels to a JSON list.

Usage:
  python -m food_analyzer.classification.dump_labels \
    --config config.json \
    --out ingredient_labels.json

This uses your config to initialize FoodClassifier, then writes classifier.labels
(no hard-coded labels). If labels cannot be produced, it exits with an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from food_analyzer.classification.classifier import FoodClassifier
from food_analyzer.utils.config import load_config


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export classifier ingredient labels to JSON")
    p.add_argument("--config", default=None, help="Path to your config.json/yaml (optional)")
    p.add_argument("--out", required=True, help="Path to write JSON list of ingredient labels")
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    classifier_cfg = cfg.get("classifier", {})
    models_cfg = cfg.get("models", {})
    clip_cfg = models_cfg.get("clip", {})

    dynamic_labels_source = classifier_cfg.get("dynamic_labels_source")
    intelligent_labels_method = classifier_cfg.get("intelligent_labels_method")

    try:
        clf = FoodClassifier(
            device=classifier_cfg.get("device"),
            backend=str(classifier_cfg.get("backend", "efficientnet_b0")),
            dynamic_labels_source=dynamic_labels_source,
            intelligent_labels_method=intelligent_labels_method,
            temperature=classifier_cfg.get("temperature", 1.0),
            confidence_threshold=classifier_cfg.get("confidence_threshold", 0.3),
            multi_scale=classifier_cfg.get("multi_scale", False),
            ensemble_weights=classifier_cfg.get("ensemble_weights", [1.0, 1.0, 1.0]),
            clip_model_name=clip_cfg.get("name", "ViT-L-14-336"),
            clip_pretrained=clip_cfg.get("pretrained", "openai"),
        )
    except Exception as exc:
        print(f"Failed to initialize classifier: {exc}")
        return 2

    labels = getattr(clf, "labels", None)
    if not labels:
        print("Classifier produced no labels. Ensure dynamic/intelligent label sources are configured.")
        return 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(labels), indent=2), encoding="utf-8")
    print(f"Wrote {len(labels)} labels to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
