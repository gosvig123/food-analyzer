"""Parameter optimization command."""
from __future__ import annotations

import argparse
from pathlib import Path


def run_optimization(args: argparse.Namespace) -> int:
    """Run parameter optimization based on evaluation results.
    
    Args:
        args: Namespace with evaluation_file attribute
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        from food_analyzer.optimization.parameter_optimizer import (
            optimize_from_evaluation,
        )
    except ImportError:
        print("Error: Parameter optimization module not available")
        return 1

    # Find evaluation file
    if getattr(args, "evaluation_file", None):
        eval_path = Path(args.evaluation_file)
    else:
        # Find latest evaluation file in results
        results_dir = Path("results")
        eval_files = list(results_dir.glob("ground_truth_evaluation.json"))
        if not eval_files:
            print(
                "No evaluation files found. Run inference first to generate evaluation data."
            )
            return 1
        eval_path = max(eval_files, key=lambda p: p.stat().st_mtime)

    if not eval_path.exists():
        print(f"Evaluation file not found: {eval_path}")
        return 1

    print(f"Running optimization based on: {eval_path}")

    # Run optimization
    try:
        optimize_from_evaluation(str(eval_path))
        return 0
    except Exception as e:
        print(f"Parameter optimization failed: {e}")
        return 1


def run_optimization_from_path(eval_path: str | Path) -> int:
    """Run optimization from a specific evaluation file path.
    
    Args:
        eval_path: Path to evaluation JSON file
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    class Args:
        evaluation_file = str(eval_path)
    
    return run_optimization(Args())
