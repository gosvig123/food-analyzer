"""
Parameter-based optimization system for food analyzer.

This module optimizes fundamental parameters that affect the entire detection pipeline,
rather than hardcoded per-ingredient adjustments.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class OptimizationTarget:
    """Target metrics for optimization."""

    recall_target: float = 0.90  # Primary goal: high recall
    precision_min: float = 0.10  # Minimum acceptable precision
    f1_weight_recall: float = 0.8  # Weight recall higher in F1 calculation


class ParameterOptimizer:
    """Optimize fundamental detection and classification parameters."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.target = OptimizationTarget()

    def load_config(self) -> Dict[str, Any]:
        """Load current configuration."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save optimized configuration."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def apply_parameters(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply parameter values to config using dot notation."""
        new_config = copy.deepcopy(config)

        for param_name, value in params.items():
            keys = param_name.split(".")
            current = new_config

            # Navigate to the right nested location
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the final value
            current[keys[-1]] = value

        return new_config

    def get_parameter_suggestions(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate parameter suggestions based on evaluation results."""
        current_metrics = evaluation_results.get("overall_metrics", {})
        current_recall = current_metrics.get("average_recall", 0.0)
        current_precision = current_metrics.get("average_precision", 0.0)

        suggestions = {}

        # If recall is too low, be more aggressive
        if current_recall < self.target.recall_target:
            recall_gap = self.target.recall_target - current_recall

            if recall_gap > 0.3:  # Very low recall
                suggestions.update(
                    {
                        "detector.score_threshold": 0.001,  # Ultra-low threshold
                        "detector.max_detections": 200,
                        "detector.iou_threshold": 0.3,  # Allow more overlaps
                        "classifier.temperature": 0.8,  # Less conservative
                        "classifier.confidence_threshold": 0.1,
                    }
                )
            elif recall_gap > 0.15:  # Moderately low recall
                suggestions.update(
                    {
                        "detector.score_threshold": 0.005,
                        "detector.max_detections": 150,
                        "detector.iou_threshold": 0.4,
                        "classifier.temperature": 1.0,
                        "classifier.confidence_threshold": 0.2,
                    }
                )
            else:  # Fine-tuning needed
                suggestions.update(
                    {
                        "detector.score_threshold": 0.01,
                        "detector.max_detections": 100,
                        "detector.iou_threshold": 0.5,
                        "classifier.temperature": 1.2,
                        "classifier.confidence_threshold": 0.3,
                    }
                )

        # If precision is acceptable but recall needs improvement
        if (
            current_precision >= self.target.precision_min
            and current_recall < self.target.recall_target
        ):
            suggestions["classifier.multi_scale"] = True
            suggestions["detector.augment"] = True
            suggestions["detector.imgsz"] = 1536

        return suggestions

    def optimize_parameters(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main optimization function - suggests parameter improvements."""

        # Get intelligent parameter suggestions based on current performance
        suggested_params = self.get_parameter_suggestions(evaluation_results)

        # Load current config and apply suggestions
        current_config = self.load_config()
        optimized_config = self.apply_parameters(current_config, suggested_params)

        # Prepare optimization report
        current_metrics = evaluation_results.get("overall_metrics", {})

        optimization_report = {
            "current_performance": current_metrics,
            "target_performance": {
                "recall_target": self.target.recall_target,
                "precision_minimum": self.target.precision_min,
            },
            "parameter_changes": suggested_params,
            "optimization_strategy": self._get_strategy_explanation(current_metrics),
            "expected_impact": self._predict_impact(suggested_params),
        }

        return {
            "optimized_config": optimized_config,
            "report": optimization_report,
            "should_apply": len(suggested_params) > 0,
        }

    def _get_strategy_explanation(self, metrics: Dict[str, float]) -> str:
        """Explain the optimization strategy based on current metrics."""
        recall = metrics.get("average_recall", 0.0)
        precision = metrics.get("average_precision", 0.0)

        if recall < 0.6:
            return "AGGRESSIVE_RECALL: Very low recall - using ultra-low thresholds and maximum detections"
        elif recall < 0.8:
            return "MODERATE_RECALL: Below target recall - lowering thresholds and enabling augmentation"
        elif precision < self.target.precision_min:
            return "PRECISION_BALANCE: Good recall but precision too low - slight threshold increase"
        else:
            return "FINE_TUNING: Near optimal - minor adjustments for better balance"

    def _predict_impact(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Predict the impact of parameter changes."""
        impact = {}

        for param, value in params.items():
            if "score_threshold" in param:
                if value <= 0.005:
                    impact[param] = "HIGH recall increase, MEDIUM precision decrease"
                else:
                    impact[param] = "MEDIUM recall increase, LOW precision decrease"

            elif "max_detections" in param:
                if value >= 150:
                    impact[param] = "HIGH recall increase, potential processing cost"
                else:
                    impact[param] = "MEDIUM recall increase"

            elif "temperature" in param:
                if value < 1.0:
                    impact[param] = "More confident predictions, better recall"
                else:
                    impact[param] = "More conservative predictions, better precision"

            elif "multi_scale" in param and value:
                impact[param] = "MEDIUM recall increase, HIGHER processing cost"

        return impact


def optimize_from_evaluation(
    evaluation_file: str = "results/ground_truth_evaluation.json",
) -> None:
    """Main entry point for parameter optimization."""

    # Load evaluation results
    eval_path = Path(evaluation_file)
    if not eval_path.exists():
        print(f"Evaluation file not found: {evaluation_file}")
        return

    with open(eval_path, "r") as f:
        evaluation_results = json.load(f)

    # Run optimization
    optimizer = ParameterOptimizer()
    result = optimizer.optimize_parameters(evaluation_results)

    if result["should_apply"]:
        # Save the optimized config
        optimizer.save_config(result["optimized_config"])

        # Save optimization report
        report_path = Path("results/parameter_optimization.json")
        with open(report_path, "w") as f:
            json.dump(result["report"], f, indent=2)

        print("============================================================")
        print("PARAMETER OPTIMIZATION COMPLETE")
        print("============================================================")
        print()
        print("Current Performance:")
        current = result["report"]["current_performance"]
        print(f"  Precision: {current.get('average_precision', 0):.3f}")
        print(f"  Recall:    {current.get('average_recall', 0):.3f}")
        print(f"  F1 Score:  {current.get('average_f1', 0):.3f}")
        print()
        print("Parameter Changes Applied:")
        for param, value in result["report"]["parameter_changes"].items():
            impact = result["report"]["expected_impact"].get(param, "")
            print(f"  {param}: {value} → {impact}")
        print()
        print(f"Strategy: {result['report']['optimization_strategy']}")
        print()
        print("Optimization results saved to: results/parameter_optimization.json")
        print("Config updated with optimized parameters!")
        print()
        print("Run inference again to see the improvements!")

    else:
        print("No parameter optimization needed - performance is optimal.")


if __name__ == "__main__":
    optimize_from_evaluation()
