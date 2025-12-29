"""CLI entry point for running food inference over images with configurable settings.

This is a simplified main module that delegates to specialized command modules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from food_analyzer.cli import compare_results, run_inference
from food_analyzer.cli.optimize import run_optimization
from food_analyzer.utils.config import load_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze food images and detect ingredients."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="data",
        help="Directory containing images to analyze",
    )
    parser.add_argument(
        "--config", dest="config", default=None, 
        help="Path to JSON/YAML config file"
    )
    parser.add_argument(
        "--no-depth", action="store_true", 
        help="Disable depth estimation stage"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Detector score threshold override",
    )
    parser.add_argument(
        "--grams-full-plate",
        type=float,
        default=None,
        help="Grams corresponding to full-frame portion",
    )
    parser.add_argument(
        "--tta",
        nargs="+",
        type=int,
        default=None,
        help="Enable detector TTA with provided shorter-edge sizes",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Where to write JSON outputs (overrides config)",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs="+",
        default=None,
        help="Compare detection counts across result folders",
    )
    parser.add_argument(
        "--compare-out",
        default=None,
        help="Optional CSV path for --compare-dirs output",
    )
    
    # Dish-first flags
    parser.add_argument(
        "--dish-first",
        action="store_true",
        help="Enable dish-first reweighting",
    )
    parser.add_argument(
        "--dish-labels",
        default=None,
        help="Path to dish labels file",
    )
    parser.add_argument(
        "--dish-priors",
        default=None,
        help="Path to dish->ingredient priors file",
    )
    parser.add_argument(
        "--dish-alpha", 
        type=float, 
        default=None, 
        help="Reweighting strength alpha"
    )
    parser.add_argument(
        "--dish-min-secondary-conf", 
        type=float, 
        default=None, 
        help="Min confidence for secondary candidates"
    )
    parser.add_argument(
        "--dish-max-secondary", 
        type=int, 
        default=None, 
        help="Max secondary candidates per detection"
    )
    
    # Mode flags
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run parameter optimization based on evaluation results",
    )
    parser.add_argument(
        "--maximize-recall",
        action="store_true",
        help="Enable high-recall mode with looser thresholds",
    )
    parser.add_argument(
        "--evaluation-file",
        default=None,
        help="Evaluation file for optimization",
    )
    
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply command line overrides to configuration."""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    
    if args.no_depth:
        cfg.setdefault("depth", {})["enabled"] = False
        
    if args.score_threshold is not None:
        cfg.setdefault("detector", {})["score_threshold"] = float(args.score_threshold)
        
    if getattr(args, "maximize_recall", False):
        det = cfg.setdefault("detector", {})
        det["score_threshold"] = min(0.05, float(det.get("score_threshold", 0.5)))
        det["max_detections"] = max(300, int(det.get("max_detections", 100)))
        det["augment"] = True
        cls = cfg.setdefault("classifier", {})
        cls["confidence_threshold"] = 0.0
        cfg.setdefault("pipeline", {})["maximize_recall"] = True

    # Dish-first overrides
    if getattr(args, "dish_first", False):
        cfg.setdefault("dish_first", {})["enabled"] = True
    if getattr(args, "dish_labels", None):
        cfg.setdefault("dish_first", {}).setdefault("dish_classifier", {})["labels_path"] = str(args.dish_labels)
    if getattr(args, "dish_priors", None):
        pr = cfg.setdefault("dish_first", {}).setdefault("priors", {})
        pr["source"] = "file"
        pr["path"] = str(args.dish_priors)
    if getattr(args, "dish_alpha", None) is not None:
        cfg.setdefault("dish_first", {}).setdefault("reweighting", {})["alpha"] = float(args.dish_alpha)
    if getattr(args, "dish_min_secondary_conf", None) is not None:
        cfg.setdefault("dish_first", {}).setdefault("reweighting", {})["min_secondary_conf"] = float(args.dish_min_secondary_conf)
    if getattr(args, "dish_max_secondary", None) is not None:
        cfg.setdefault("dish_first", {}).setdefault("reweighting", {})["max_secondary"] = int(args.dish_max_secondary)
        
    if args.grams_full_plate is not None:
        cfg.setdefault("volume", {})["grams_for_full_plate"] = float(args.grams_full_plate)
    if args.results_dir is not None:
        cfg.setdefault("io", {})["results_dir"] = str(args.results_dir)
    if getattr(args, "tta", None):
        det = cfg.setdefault("detector", {})
        det["tta_imgsz"] = list(map(int, args.tta))
        det["augment"] = True
        
    return cfg


def main(argv: list[str]) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Compare mode
    if args.compare_dirs:
        compare_results(args.compare_dirs, out_path=args.compare_out)
        return 0

    # Optimization mode
    if args.optimize:
        return run_optimization(args)

    # Inference mode
    target = Path(args.directory)
    if not target.exists():
        print(f"Directory not found: {target}")
        return 1

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    run_inference(target, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
