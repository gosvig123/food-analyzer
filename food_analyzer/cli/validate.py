"""Ground truth validation utilities."""
from __future__ import annotations

import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from food_analyzer.utils.labels import LabelNormalizer


def load_ground_truth(ground_truth_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth ingredients per plate type from JSON or CSV.
    
    Args:
        ground_truth_path: Path to ground truth file
        
    Returns:
        Dict mapping plate type to set of expected ingredients
    """
    ground_truth: Dict[str, Set[str]] = {}

    if ground_truth_path.suffix.lower() == ".json":
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for plate_type, ingredients in data.items():
            ground_truth[plate_type] = set(
                ingredient.lower() for ingredient in ingredients
            )
        return ground_truth

    else:
        # Legacy CSV format support
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return ground_truth

        plate_types = [cell.strip() for cell in rows[1][1:] if cell.strip()]

        for plate_type in plate_types:
            ground_truth[plate_type] = set()

        for row in rows[2:]:
            if not row or not row[0].strip().startswith("Ingredient"):
                continue
            ingredients = [cell.strip().lower() for cell in row[1:] if cell.strip()]

            for i, ingredient in enumerate(ingredients):
                if i < len(plate_types) and ingredient:
                    ground_truth[plate_types[i]].add(ingredient)

        return ground_truth


def validate_against_ground_truth(
    detections: List[dict],
    image_name: str,
    ground_truth: Dict[str, Set[str]],
    label_normalizer: LabelNormalizer | None = None,
) -> Dict[str, Any]:
    """Validate detections against ground truth with fuzzy plate-type matching.
    
    Args:
        detections: List of detection dicts with 'label' key
        image_name: Name of the image file
        ground_truth: Dict mapping plate type to expected ingredients
        label_normalizer: Optional normalizer for label matching
        
    Returns:
        Dict with precision, recall, f1, and detailed results
    """
    image_stem = Path(image_name).stem
    plate_type: str | None = None

    # 1) Direct substring match
    for gt_plate in ground_truth.keys():
        if gt_plate in image_stem:
            plate_type = gt_plate
            break

    # 2) Fuzzy match if not found
    if not plate_type:
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
        
        stem_n = _norm(image_stem)
        best_key = None
        best_ratio = 0.0
        
        for gt_plate in ground_truth.keys():
            r = difflib.SequenceMatcher(None, stem_n, _norm(gt_plate)).ratio()
            if r > best_ratio:
                best_ratio = r
                best_key = gt_plate
        
        if best_key is not None and best_ratio >= 0.55:
            plate_type = best_key

    if not plate_type:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "plate_type": "unknown",
            "expected": [],
            "detected": [],
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    expected_ingredients = ground_truth[plate_type]
    if not expected_ingredients:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "plate_type": plate_type,
            "expected": [],
            "detected": [],
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    # Get detected ingredients
    detected_ingredients: Set[str] = set()
    for detection in detections:
        raw_label = detection.get("label")
        if label_normalizer is not None:
            normalized = label_normalizer.normalize(raw_label)
        else:
            normalized = str(raw_label).strip().lower() if raw_label else None
        if normalized:
            detected_ingredients.add(normalized)

    # Calculate metrics
    true_positives = len(detected_ingredients.intersection(expected_ingredients))
    precision = (
        true_positives / len(detected_ingredients) if detected_ingredients else 0.0
    )
    recall = true_positives / len(expected_ingredients) if expected_ingredients else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "plate_type": plate_type,
        "expected": (
            [label_normalizer.display(item) for item in sorted(expected_ingredients)]
            if label_normalizer
            else sorted(expected_ingredients)
        ),
        "detected": (
            [label_normalizer.display(item) for item in sorted(detected_ingredients)]
            if label_normalizer
            else sorted(detected_ingredients)
        ),
        "true_positives": true_positives,
        "false_positives": len(detected_ingredients) - true_positives,
        "false_negatives": len(expected_ingredients) - true_positives,
    }


def print_ground_truth_report(validation_results: List[Dict]) -> None:
    """Print comprehensive ground truth validation report.
    
    Args:
        validation_results: List of validation result dicts
    """
    if not validation_results:
        return

    print("\n" + "=" * 60)
    print("GROUND TRUTH VALIDATION REPORT")
    print("=" * 60)

    # Overall metrics
    total_precision = sum(r["precision"] for r in validation_results)
    total_recall = sum(r["recall"] for r in validation_results)
    total_f1 = sum(r["f1"] for r in validation_results)
    num_images = len(validation_results)

    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_f1 = total_f1 / num_images

    print(f"\nOverall Performance:")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall:    {avg_recall:.3f}")
    print(f"Average F1 Score:  {avg_f1:.3f}")

    # Per plate type breakdown
    plate_metrics: Dict[str, List[Dict]] = defaultdict(list)
    for result in validation_results:
        plate_type = result["plate_type"]
        if plate_type != "unknown":
            plate_metrics[plate_type].append(result)

    print(f"\nPer Plate Type Performance:")
    print("-" * 50)
    for plate_type, results in plate_metrics.items():
        if not results:
            continue
        avg_p = sum(r["precision"] for r in results) / len(results)
        avg_r = sum(r["recall"] for r in results) / len(results)
        avg_f = sum(r["f1"] for r in results) / len(results)
        print(f"{plate_type:15} | P: {avg_p:.3f} | R: {avg_r:.3f} | F1: {avg_f:.3f}")

    # Detailed per-image results
    print(f"\nDetailed Results:")
    print("-" * 50)
    for result in validation_results:
        image_name = result.get("image_name", "unknown")
        print(f"\n{image_name} ({result['plate_type']}):")
        print(
            f"  P: {result['precision']:.3f} | R: {result['recall']:.3f} | F1: {result['f1']:.3f}"
        )
        print(f"  Expected: {', '.join(result['expected'])}")
        print(f"  Detected: {', '.join(result['detected'])}")
        if result["false_negatives"] > 0:
            missed = set(result["expected"]) - set(result["detected"])
            print(f"  Missed: {', '.join(missed)}")

    print("=" * 60)
