"""CLI entry point for running food inference over images with configurable settings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from PIL import Image, ImageDraw, UnidentifiedImageError

from food_analyzer.classification.classifier import FoodClassifier
from food_analyzer.core.pipeline import FoodInferencePipeline
from food_analyzer.detection.depth import DepthEstimator
from food_analyzer.detection.detector import FoodDetector
from food_analyzer.utils.config import load_config, resolve_path_relative_to_project


def iter_image_paths(directory: Path, extensions: Set[str]) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def format_result_row(index: int, result: dict) -> str:
    label = result["label"]
    confidence = float(result["confidence"])
    return f"{index:>2} | {label:<20} | {confidence:>6.2f}"


def aggregate_results(results: List[dict]) -> List[dict]:
    summary: dict[str, dict[str, float]] = {}
    for item in results:
        label = str(item["label"])
        entry = summary.setdefault(
            label,
            {
                "count": 0,
                "grams": 0.0,
                "calories": 0.0,
                "max_confidence": 0.0,
            },
        )
        entry["count"] += 1
        entry["max_confidence"] = max(
            entry["max_confidence"], float(item["confidence"])
        )
    ordered = sorted(
        (
            {
                "label": label,
                "count": values["count"],
                "grams": values["grams"],
                "calories": values["calories"],
                "max_confidence": values["max_confidence"],
            }
            for label, values in summary.items()
        ),
        key=lambda item: (item["calories"], item["grams"]),
        reverse=True,
    )
    return ordered


def format_aggregate_row(entry: dict) -> str:
    return (
        f"- {entry['label']} x{entry['count']} | avg conf {entry['max_confidence']:.2f} | "
        f"{entry['grams']:.1f}g | {entry['calories']:.1f} kcal"
    )


def write_results(
    results_dir: Path,
    image_path: Path,
    detections: List[dict],
    aggregates: List[dict],
    totals: dict,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path),
        "top_detections": detections[:5],
        "detections": detections,
        "aggregated": aggregates,
        "totals": totals,
    }
    output_path = results_dir / f"{image_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_highest_predictions_csv(
    results_dir: Path,
    image_path: Path,
    detections: List[dict],
) -> None:
    """Write the highest prediction for each ingredient per image to CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "highest_predictions.csv"

    # Check if CSV file exists to determine if we need headers
    write_headers = not csv_path.exists()

    # Find highest confidence prediction for each unique ingredient
    ingredient_predictions = {}
    for detection in detections:
        label = detection.get("label", "unknown")
        confidence = float(detection.get("confidence", 0.0))

        if (
            label not in ingredient_predictions
            or confidence > ingredient_predictions[label]["confidence"]
        ):
            ingredient_predictions[label] = {
                "image": image_path.name,
                "ingredient": label,
                "confidence": confidence,
                "box": detection.get("box", []),
            }

    # Write to CSV
    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "image",
            "ingredient",
            "confidence",
            "box_left",
            "box_top",
            "box_right",
            "box_bottom",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        for prediction in ingredient_predictions.values():
            box = prediction["box"]
            row = {
                "image": prediction["image"],
                "ingredient": prediction["ingredient"],
                "confidence": prediction["confidence"],
                "box_left": box[0] if len(box) >= 4 else "",
                "box_top": box[1] if len(box) >= 4 else "",
                "box_right": box[2] if len(box) >= 4 else "",
                "box_bottom": box[3] if len(box) >= 4 else "",
            }
            writer.writerow(row)


def save_visuals(
    results_dir: Path,
    image_path: Path,
    detections: List[dict],
    save_overlays: bool,
    save_crops: bool,
    save_masks: bool,
) -> None:
    if not (save_overlays or save_crops):
        return
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return

    if save_overlays:
        overlays_dir = results_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        for det in detections:
            box = det.get("box", None)
            label = str(det.get("label", ""))
            conf = float(det.get("confidence", 0.0))
            mask_poly = det.get("mask_polygon")
            # Draw mask polygon first (semi-transparent)
            if isinstance(mask_poly, list) and len(mask_poly) >= 3:
                try:
                    poly_pts = [(int(x), int(y)) for x, y in mask_poly]
                    draw.polygon(
                        poly_pts, fill=(255, 0, 0, 80), outline=(255, 0, 0, 180)
                    )
                except Exception:
                    pass
            # Draw bounding box + label
            if box and len(box) == 4:
                left, top, right, bottom = [int(v) for v in box]
                draw.rectangle(
                    [(left, top), (right, bottom)], outline=(255, 0, 0, 220), width=3
                )
                txt = f"{label} {conf:.2f}"
                tw, th = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(txt), 12
                draw.rectangle(
                    [(left, max(0, top - th - 4)), (left + int(tw) + 6, top)],
                    fill=(255, 0, 0, 220),
                )
                draw.text(
                    (left + 3, max(0, top - th - 2)), txt, fill=(255, 255, 255, 255)
                )
        composed = Image.alpha_composite(base, overlay)
        composed.save(overlays_dir / f"{image_path.stem}_overlay.png")

    if save_crops:
        crops_dir = results_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for idx, det in enumerate(detections):
            box = det.get("box", None)
            label = str(det.get("label", "item"))
            mask_poly = det.get("mask_polygon")
            if not box or len(box) != 4:
                continue
            # Base bbox crop
            bbox_crop = image.crop(tuple(int(v) for v in box))
            # Mask-tight crop (if available)
            mask_crop = None
            if isinstance(mask_poly, list) and len(mask_poly) >= 3:
                try:
                    xs = [int(x) for x, _ in mask_poly]
                    ys = [int(y) for _, y in mask_poly]
                    pad = 4
                    left = max(min(xs) - pad, 0)
                    top = max(min(ys) - pad, 0)
                    right = min(max(xs) + pad, image.size[0])
                    bottom = min(max(ys) + pad, image.size[1])
                    mask_crop = image.crop((left, top, right, bottom))
                except Exception:
                    mask_crop = None
            # Choose which to store as primary crop (mask if available)
            primary_crop = mask_crop if mask_crop is not None else bbox_crop
            safe_label = label.replace(" ", "_")
            # Save both variants
            (crops_dir / "bbox").mkdir(parents=True, exist_ok=True)
            (crops_dir / "mask").mkdir(parents=True, exist_ok=True)
            primary_crop.save(
                crops_dir / f"{image_path.stem}_{idx:02d}_{safe_label}.jpg"
            )
            bbox_crop.save(
                crops_dir
                / "bbox"
                / f"{image_path.stem}_{idx:02d}_{safe_label}_bbox.jpg"
            )
            (mask_crop if mask_crop is not None else bbox_crop).save(
                crops_dir
                / "mask"
                / f"{image_path.stem}_{idx:02d}_{safe_label}_mask.jpg"
            )
            # Montage (side-by-side bbox vs mask-tight)
            try:
                left_img, right_img = (
                    bbox_crop,
                    (mask_crop if mask_crop is not None else bbox_crop),
                )
                h = max(left_img.height, right_img.height)
                new_left = Image.new("RGB", (left_img.width, h), (255, 255, 255))
                new_left.paste(left_img, (0, 0))
                new_right = Image.new("RGB", (right_img.width, h), (255, 255, 255))
                new_right.paste(right_img, (0, 0))
                montage = Image.new(
                    "RGB", (new_left.width + new_right.width, h), (255, 255, 255)
                )
                montage.paste(new_left, (0, 0))
                montage.paste(new_right, (new_left.width, 0))
                (crops_dir / "montage").mkdir(parents=True, exist_ok=True)
                montage.save(
                    crops_dir
                    / "montage"
                    / f"{image_path.stem}_{idx:02d}_{safe_label}_montage.jpg"
                )
            except Exception:
                pass

    if save_masks:
        masks_dir = results_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for idx, det in enumerate(detections):
            mask_poly = det.get("mask_polygon")
            if not (isinstance(mask_poly, list) and len(mask_poly) >= 3):
                continue
            try:
                mask_img = Image.new("L", image.size, 0)
                mdraw = ImageDraw.Draw(mask_img)
                poly_pts = [(int(x), int(y)) for x, y in mask_poly]
                mdraw.polygon(poly_pts, fill=255)
                label = str(det.get("label", "item")).replace(" ", "_")
                mask_img.save(
                    masks_dir / f"{image_path.stem}_{idx:02d}_{label}_mask.png"
                )
            except Exception:
                continue


def build_pipeline_from_config(cfg: dict) -> FoodInferencePipeline:
    # Components
    detector_cfg = cfg.get("detector", {})
    detector = FoodDetector(
        score_threshold=float(detector_cfg.get("score_threshold", 0.25)),
        iou_threshold=float(detector_cfg.get("iou_threshold", 0.45)),
        max_detections=int(detector_cfg.get("max_detections", 100)),
        device=detector_cfg.get("device"),
        backend=str(detector_cfg.get("backend", "mask_rcnn_deeplabv3")),
        model_name=detector_cfg.get("model_name") or None,
        imgsz=int(detector_cfg.get("imgsz"))
        if detector_cfg.get("imgsz") is not None
        else None,
        retina_masks=bool(detector_cfg.get("retina_masks", True)),
        augment=bool(detector_cfg.get("augment", False)),
        tta_imgsz=detector_cfg.get("tta_imgsz") or None,
        refine_masks=bool(detector_cfg.get("refine_masks", True)),
        refine_method=str(detector_cfg.get("refine_method", "sam")),
        morph_kernel=int(detector_cfg.get("morph_kernel", 5)),
        morph_iters=int(detector_cfg.get("morph_iters", 2)),
        sam_checkpoint=detector_cfg.get("sam_checkpoint"),
        sam_model_type=detector_cfg.get("sam_model_type"),
        fusion_method=str(detector_cfg.get("fusion_method", "soft_nms")),
        soft_nms_sigma=float(detector_cfg.get("soft_nms_sigma", 0.5)),
    )

    classifier_cfg = cfg.get("classifier", {})
    dynamic_labels_source = classifier_cfg.get("dynamic_labels_source")
    intelligent_labels_method = classifier_cfg.get("intelligent_labels_method")
    classifier = FoodClassifier(
        device=classifier_cfg.get("device"),
        backend=str(classifier_cfg.get("backend", "efficientnet_b0")),
        dynamic_labels_source=dynamic_labels_source,
        intelligent_labels_method=intelligent_labels_method,
        temperature=classifier_cfg.get("temperature", 1.0),
        confidence_threshold=classifier_cfg.get("confidence_threshold", 0.3),
        multi_scale=classifier_cfg.get("multi_scale", False),
        ensemble_weights=classifier_cfg.get("ensemble_weights", [1.0, 1.0, 1.0]),
    )

    depth_enabled = bool(cfg.get("depth", {}).get("enabled", True))
    depth_estimator = DepthEstimator() if depth_enabled else None

    pipeline_cfg = cfg.get("pipeline", {})
    return FoodInferencePipeline(
        detector=detector,
        classifier=classifier,
        depth_estimator=depth_estimator,
        use_detector_labels=bool(pipeline_cfg.get("use_detector_labels", False)),
    )


def run_inference(target_dir: Path, cfg: dict) -> None:
    pipeline = build_pipeline_from_config(cfg)

    exts = {
        e.lower()
        for e in cfg.get("io", {}).get("image_extensions", [".jpg", ".jpeg", ".png"])
    }
    results_dir = Path(cfg.get("io", {}).get("results_dir", "results"))
    save_overlays = bool(cfg.get("io", {}).get("save_overlays", False))
    save_crops = bool(cfg.get("io", {}).get("save_crops", False))
    save_masks = bool(cfg.get("io", {}).get("save_masks", False))

    images = list(iter_image_paths(target_dir, exts))
    if not images:
        print(f"No images found in {target_dir}")
        return

    # Load ground truth if available
    # Try JSON first, fall back to CSV for compatibility
    ground_truth_path = Path("ground_truth.json")
    if not ground_truth_path.exists():
        ground_truth_path = Path("ground_truth.csv")
    ground_truth = {}
    validation_results = []

    if ground_truth_path.exists():
        try:
            ground_truth = load_ground_truth(ground_truth_path)
            print(f"Loaded ground truth for {len(ground_truth)} plate types")
        except Exception as exc:
            print(f"Failed to load ground truth: {exc}")

    for image_path in images:
        try:
            results = [asdict(item) for item in pipeline.analyze(image_path)]
        except UnidentifiedImageError as exc:
            print(f"\n=== {image_path} ===")
            print(f"Skipped file (not a valid image): {exc}")
            continue
        if not results:
            print(f"\n=== {image_path} ===")
            print("No detections")
            continue

        results.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        print(f"\n=== {image_path} ===")
        print("Top detections (sorted by confidence):")
        print(" # | Label                |  Conf. | Portion | Calories")
        print("-- + -------------------- + ------- + ------- + --------")
        for idx, result in enumerate(results[:5], start=1):
            print(format_result_row(idx, result))

        aggregates = aggregate_results(results)
        print("\nAggregated summary:")
        for entry in aggregates:
            print(format_aggregate_row(entry))

        total_calories = sum(entry["calories"] for entry in aggregates)
        total_grams = sum(entry["grams"] for entry in aggregates)
        print(f"\nTotals: {total_grams:.1f}g | {total_calories:.1f} kcal")

        # Save visualization assets if enabled
        save_visuals(
            results_dir=results_dir,
            image_path=image_path,
            detections=results,
            save_overlays=save_overlays,
            save_crops=save_crops,
            save_masks=save_masks,
        )

        write_results(
            results_dir=results_dir,
            image_path=image_path,
            detections=results,
            aggregates=aggregates,
            totals={"grams": total_grams, "calories": total_calories},
        )

        # Write highest predictions to CSV
        write_highest_predictions_csv(
            results_dir=results_dir,
            image_path=image_path,
            detections=results,
        )

        # Validate against ground truth if available
        if ground_truth:
            validation_result = validate_against_ground_truth(
                detections=results,
                image_name=image_path.name,
                ground_truth=ground_truth,
            )
            validation_result["image_name"] = image_path.name
            validation_results.append(validation_result)

    # Save and print ground truth validation report
    if validation_results:
        save_ground_truth_evaluation(results_dir, validation_results)
        copy_analysis_to_results(results_dir)
        print_ground_truth_report(validation_results)


def load_ground_truth(ground_truth_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth ingredients per plate type from JSON or CSV."""
    ground_truth = {}

    if ground_truth_path.suffix.lower() == ".json":
        # Load from JSON format
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert lists to sets and normalize to lowercase
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

        # Skip empty first row and get plate types from second row
        if len(rows) < 2:
            return ground_truth

        plate_types = [cell.strip() for cell in rows[1][1:] if cell.strip()]

        # Initialize ground truth dict
        for plate_type in plate_types:
            ground_truth[plate_type] = set()

        # Process ingredient rows
        for row in rows[2:]:
            if not row or not row[0].strip().startswith("Ingredient"):
                continue
            ingredients = [cell.strip().lower() for cell in row[1:] if cell.strip()]

            for i, ingredient in enumerate(ingredients):
                if i < len(plate_types) and ingredient:
                    ground_truth[plate_types[i]].add(ingredient)

        return ground_truth


def validate_against_ground_truth(
    detections: List[dict], image_name: str, ground_truth: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Validate detections against ground truth for an image."""
    # Extract plate type from image name (assuming format: plateType_*.jpg)
    image_stem = Path(image_name).stem
    plate_type = None

    for gt_plate in ground_truth.keys():
        if gt_plate in image_stem:
            plate_type = gt_plate
            break

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

    # Get detected ingredients (normalize to lowercase)
    detected_ingredients = set()
    for detection in detections:
        label = detection.get("label", "").lower()
        if label:
            # Load dynamic synonyms and combine with base synonyms
            synonyms = load_dynamic_synonyms()

            # Base synonym mappings
            base_synonyms = {
                "bell pepper": "green pepper",
                "green bell pepper": "green pepper",
                "red bell pepper": "green pepper",
                "pepper": "green pepper",
                "egg": "eggs",
                "soya sauce": "soy sauce",
                "salad dressing": "dressing",
                "ranch dressing": "dressing",
                "vinaigrette": "dressing",
                "nori": "seaweed",
                "kelp": "seaweed",
                "pasta": "noodles",
                "ramen": "noodles",
                "udon": "noodles",
                "sweet corn": "corn",
                "lime juice": "lime",
                "lemon": "lime",  # Similar citrus
                "vegetable": "",  # filter out generic terms
                "food": "",
                "dish": "",
                "meal": "",
            }

            # Merge dynamic synonyms with base synonyms
            synonyms.update(base_synonyms)
            label = synonyms.get(label, label)
            if label:  # only add non-empty labels
                detected_ingredients.add(label)

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
        "expected": list(expected_ingredients),
        "detected": list(detected_ingredients),
        "true_positives": true_positives,
        "false_positives": len(detected_ingredients) - true_positives,
        "false_negatives": len(expected_ingredients) - true_positives,
    }


def save_ground_truth_evaluation(
    results_dir: Path, validation_results: List[Dict]
) -> None:
    """Save ground truth evaluation results to results folder."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate overall metrics
    total_precision = sum(r["precision"] for r in validation_results)
    total_recall = sum(r["recall"] for r in validation_results)
    total_f1 = sum(r["f1"] for r in validation_results)
    num_images = len(validation_results)

    avg_precision = total_precision / num_images if num_images > 0 else 0.0
    avg_recall = total_recall / num_images if num_images > 0 else 0.0
    avg_f1 = total_f1 / num_images if num_images > 0 else 0.0

    # Per plate type breakdown
    plate_metrics = defaultdict(list)
    for result in validation_results:
        plate_type = result["plate_type"]
        if plate_type != "unknown":
            plate_metrics[plate_type].append(result)

    # Create evaluation report
    evaluation_report = {
        "overall_metrics": {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "total_images": num_images,
        },
        "per_plate_type": {},
        "detailed_results": validation_results,
    }

    # Add per-plate metrics
    for plate_type, results in plate_metrics.items():
        if results:
            avg_p = sum(r["precision"] for r in results) / len(results)
            avg_r = sum(r["recall"] for r in results) / len(results)
            avg_f = sum(r["f1"] for r in results) / len(results)
            evaluation_report["per_plate_type"][plate_type] = {
                "precision": avg_p,
                "recall": avg_r,
                "f1": avg_f,
                "num_images": len(results),
            }

    # Save to JSON file
    eval_path = results_dir / "ground_truth_evaluation.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    # Also save as CSV for easy analysis
    csv_path = results_dir / "ground_truth_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "image_name",
            "plate_type",
            "precision",
            "recall",
            "f1",
            "true_positives",
            "false_positives",
            "false_negatives",
            "expected_ingredients",
            "detected_ingredients",
            "missed_ingredients",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in validation_results:
            missed = set(result["expected"]) - set(result["detected"])
            writer.writerow(
                {
                    "image_name": result["image_name"],
                    "plate_type": result["plate_type"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "true_positives": result["true_positives"],
                    "false_positives": result["false_positives"],
                    "false_negatives": result["false_negatives"],
                    "expected_ingredients": ", ".join(result["expected"]),
                    "detected_ingredients": ", ".join(result["detected"]),
                    "missed_ingredients": ", ".join(missed),
                }
            )

    print(f"Ground truth evaluation saved to:")
    print(f"  - {eval_path}")
    print(f"  - {csv_path}")


def load_dynamic_synonyms() -> dict:
    """Load dynamically discovered synonyms."""
    synonym_path = Path("dynamic_synonyms.json")
    if synonym_path.exists():
        try:
            with open(synonym_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def copy_analysis_to_results(results_dir: Path) -> None:
    """Copy ground truth analysis to results folder for reference."""
    analysis_source = Path("ground_truth_analysis.md")
    if analysis_source.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        analysis_dest = results_dir / "ground_truth_analysis.md"
        import shutil

        shutil.copy2(analysis_source, analysis_dest)
        print(f"Analysis report copied to: {analysis_dest}")


def run_optimization(args: argparse.Namespace) -> int:
    """Run parameter optimization based on evaluation results."""
    try:
        from food_analyzer.optimization.parameter_optimizer import (
            optimize_from_evaluation,
        )
    except ImportError:
        print("Error: Parameter optimization module not available")
        return 1

    # Find evaluation file
    if args.evaluation_file:
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
        # Run parameter optimization (handles its own output and file saving)
        optimize_from_evaluation(str(eval_path))
        return 0
    except Exception as e:
        print(f"Parameter optimization failed: {e}")
        return 1


def print_ground_truth_report(validation_results: List[Dict]) -> None:
    """Print comprehensive ground truth validation report."""
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
    plate_metrics = defaultdict(list)
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


def compare_results(dirs: list[str], out_path: str | None = None) -> None:
    paths = [Path(d) for d in dirs]
    # Map: image_stem -> {dir_name: count}
    per_image: dict[str, dict[str, int]] = {}
    dir_names = [p.name for p in paths]
    for p, name in zip(paths, dir_names):
        for jf in sorted(p.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            stem = Path(data.get("image", jf)).stem
            dets = data.get("detections", [])
            per_image.setdefault(stem, {})[name] = int(len(dets))
    # Print simple table
    headers = ["image"] + dir_names
    print("\n=== Comparison: detections per image ===")
    print(" | ".join(h.ljust(20) for h in headers))
    print("-+-".join(["-" * 20 for _ in headers]))
    rows: list[list[str]] = []
    for stem in sorted(per_image.keys()):
        row = [stem] + [str(per_image[stem].get(name, 0)) for name in dir_names]
        rows.append(row)
        print(" | ".join([stem.ljust(20)] + [c.ljust(20) for c in row[1:]]))
    # Optional CSV
    if out_path:
        import csv

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(r)
        print(f"\nSaved CSV: {out_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
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
        "--config", dest="config", default=None, help="Path to JSON/YAML config file"
    )
    parser.add_argument(
        "--no-depth", action="store_true", help="Disable depth estimation stage"
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
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run dynamic optimization based on latest evaluation results",
    )
    parser.add_argument(
        "--evaluation-file",
        default=None,
        help="Specific evaluation file to optimize from (default: latest in results)",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    if args.no_depth:
        cfg.setdefault("depth", {})["enabled"] = False
    if args.score_threshold is not None:
        cfg.setdefault("detector", {})["score_threshold"] = float(args.score_threshold)
    if args.grams_full_plate is not None:
        cfg.setdefault("volume", {})["grams_for_full_plate"] = float(
            args.grams_full_plate
        )
    if args.results_dir is not None:
        cfg.setdefault("io", {})["results_dir"] = str(args.results_dir)
    return cfg


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.compare_dirs:
        compare_results(args.compare_dirs, out_path=args.compare_out)
        return 0

    if args.optimize:
        return run_optimization(args)

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
