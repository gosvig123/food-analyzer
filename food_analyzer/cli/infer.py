"""Main inference command for food analysis."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from PIL import UnidentifiedImageError

from food_analyzer.classification.classifier import FoodClassifier
from food_analyzer.core.pipeline import FoodInferencePipeline
from food_analyzer.detection.depth import DepthEstimator
from food_analyzer.detection.detector import FoodDetector
from food_analyzer.io.results_writer import ResultsWriter
from food_analyzer.nutrition import NutritionEstimator
from food_analyzer.utils.config import resolve_path_relative_to_project
from food_analyzer.utils.ingredient_filter import IngredientFilter
from food_analyzer.utils.labels import (
    LabelNormalizer,
    align_ground_truth_with_labels,
    load_synonym_map,
)

from .validate import (
    load_ground_truth,
    print_ground_truth_report,
    validate_against_ground_truth,
)


def iter_image_paths(directory: Path, extensions: Set[str]) -> Iterable[Path]:
    """Iterate over image files in a directory."""
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def format_result_row(index: int, result: dict) -> str:
    """Format a single detection result for console output."""
    label = result["label"]
    confidence = float(result["confidence"])
    return f"{index:>2} | {label:<20} | {confidence:>6.2f}"


def format_aggregate_row(entry: dict) -> str:
    """Format an aggregated result for console output."""
    return (
        f"- {entry['label']} x{entry['count']} | avg conf {entry['max_confidence']:.2f} | "
        f"{entry['grams']:.1f}g | {entry['calories']:.1f} kcal"
    )


def aggregate_results(results: List[dict]) -> List[dict]:
    """Aggregate detection results by label."""
    summary: Dict[str, Dict[str, float]] = {}
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


def build_pipeline_from_config(cfg: dict) -> FoodInferencePipeline:
    """Build inference pipeline from configuration dict."""
    import json
    
    # Detector
    detector_cfg = cfg.get("detector", {})
    sam_checkpoint = resolve_path_relative_to_project(
        detector_cfg.get("sam_checkpoint")
    )
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
        sam_checkpoint=str(sam_checkpoint) if sam_checkpoint else None,
        sam_model_type=detector_cfg.get("sam_model_type"),
        fusion_method=str(detector_cfg.get("fusion_method", "soft_nms")),
        soft_nms_sigma=float(detector_cfg.get("soft_nms_sigma", 0.5)),
        semantic_food_classes=detector_cfg.get("semantic_food_classes"),
    )

    # Classifier
    classifier_cfg = cfg.get("classifier", {})
    models_cfg = cfg.get("models", {})
    clip_cfg = models_cfg.get("clip", {})
    pipeline_cfg = cfg.get("pipeline", {})

    # Build extra_labels from ground_truth.json
    extra_labels: List[str] = []
    try:
        gt_path = Path("ground_truth.json")
        if gt_path.exists():
            with gt_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                tmp: Set[str] = set()
                for items in data.values():
                    if isinstance(items, list):
                        for it in items:
                            s = str(it).strip()
                            if s:
                                tmp.add(s)
                extra_labels = sorted(tmp)
    except Exception:
        extra_labels = []

    classifier = FoodClassifier(
        device=classifier_cfg.get("device"),
        backend=str(classifier_cfg.get("backend", "efficientnet_b0")),
        dynamic_labels_source=classifier_cfg.get("dynamic_labels_source"),
        intelligent_labels_method=classifier_cfg.get("intelligent_labels_method"),
        temperature=classifier_cfg.get("temperature", 1.0),
        confidence_threshold=classifier_cfg.get("confidence_threshold", 0.3),
        multi_scale=classifier_cfg.get("multi_scale", False),
        ensemble_weights=classifier_cfg.get("ensemble_weights", [1.0, 1.0, 1.0]),
        clip_model_name=clip_cfg.get("name", "ViT-L-14-336"),
        clip_pretrained=clip_cfg.get("pretrained", "openai"),
        maximize_recall=bool(pipeline_cfg.get("maximize_recall", False)),
        extra_labels=extra_labels,
        prompts_path=classifier_cfg.get("prompts_path"),
    )

    # Depth estimator
    depth_enabled = bool(cfg.get("depth", {}).get("enabled", True))
    depth_estimator = DepthEstimator() if depth_enabled else None

    return FoodInferencePipeline(
        detector=detector,
        classifier=classifier,
        depth_estimator=depth_estimator,
        use_detector_labels=bool(pipeline_cfg.get("use_detector_labels", False)),
        maximize_recall=bool(pipeline_cfg.get("maximize_recall", False)),
    )


def run_inference(target_dir: Path, cfg: dict) -> None:
    """Run food inference on images in a directory.
    
    Args:
        target_dir: Directory containing images
        cfg: Configuration dictionary
    """
    pipeline = build_pipeline_from_config(cfg)

    # Setup label normalization
    label_normalizer: LabelNormalizer | None = None
    classifier = getattr(pipeline, "classifier", None)
    ingredient_label_list: List[str] = []
    if getattr(classifier, "labels", None):
        synonym_map = load_synonym_map()
        label_normalizer = LabelNormalizer.from_labels(classifier.labels, synonym_map)
        ingredient_label_list = list(classifier.labels)

    # Dish-first setup
    dish_processor = None
    try:
        from food_analyzer.dish_first import create_dish_first_processor
        dish_processor = create_dish_first_processor(cfg, ingredient_label_list)
    except Exception as exc:
        print(f"Dish-first processor unavailable: {exc}. Continuing without dish-first.")

    # Nutrition estimation
    nutrition_estimator = NutritionEstimator(
        default_portion_grams=float(cfg.get("volume", {}).get("grams_for_full_plate", 350.0)) / 3
    )

    # IO config
    exts = {
        e.lower()
        for e in cfg.get("io", {}).get("image_extensions", [".jpg", ".jpeg", ".png"])
    }
    results_dir = Path(cfg.get("io", {}).get("results_dir", "results"))
    
    writer = ResultsWriter(
        results_dir=results_dir,
        save_overlays=bool(cfg.get("io", {}).get("save_overlays", False)),
        save_crops=bool(cfg.get("io", {}).get("save_crops", False)),
        save_masks=bool(cfg.get("io", {}).get("save_masks", False)),
    )

    ingredient_filter = IngredientFilter()

    # Find images
    images = list(iter_image_paths(target_dir, exts))
    if not images:
        print(f"No images found in {target_dir}")
        return

    # Load ground truth
    ground_truth_path = Path("ground_truth.json")
    if not ground_truth_path.exists():
        ground_truth_path = Path("ground_truth.csv")
    
    ground_truth: Dict[str, Set[str]] = {}
    validation_results: List[dict] = []

    if ground_truth_path.exists():
        try:
            ground_truth = load_ground_truth(ground_truth_path)
            print(f"Loaded ground truth for {len(ground_truth)} plate types")
        except Exception as exc:
            print(f"Failed to load ground truth: {exc}")

    if ground_truth and label_normalizer:
        aligned_ground_truth, unmatched = align_ground_truth_with_labels(
            ground_truth, label_normalizer
        )
        if unmatched:
            print("Ground truth entries without dynamic label matches:")
            for plate, items in unmatched.items():
                formatted = ", ".join(sorted(set(items)))
                print(f"  - {plate}: {formatted}")
        ground_truth = aligned_ground_truth

    pipeline_cfg = cfg.get("pipeline", {})
    maximize_recall = bool(pipeline_cfg.get("maximize_recall", False))

    # Process each image
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

        # Filter non-ingredients
        if not maximize_recall:
            results = ingredient_filter.filter_results(results)

        # Dish-first processing
        dish_meta: dict | None = None
        if dish_processor is not None:
            dish_result = dish_processor.process(
                image=image_path,
                detections=results,
                volume_config=cfg.get("volume", {}),
            )
            results = dish_result.detections
            dish_meta = dish_result.metadata if dish_result.dish_topk else None
            if dish_result.dish_topk:
                print(dish_processor.get_top_dish_info(dish_result))

        results.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        
        # Print results
        print(f"\n=== {image_path} ===")
        print("Top detections (sorted by confidence):")
        print(" # | Label                |  Conf. | Portion | Calories")
        print("-- + -------------------- + ------- + ------- + --------")
        for idx, result in enumerate(results[:5], start=1):
            print(format_result_row(idx, result))

        # Aggregate and enrich
        aggregates = aggregate_results(results)
        if not maximize_recall:
            aggregates = ingredient_filter.filter_aggregated_results(aggregates)
        aggregates = nutrition_estimator.enrich_aggregates(aggregates)

        print("\nAggregated summary:")
        for entry in aggregates:
            print(format_aggregate_row(entry))

        total_calories = sum(entry["calories"] for entry in aggregates)
        total_grams = sum(entry["grams"] for entry in aggregates)
        total_protein = sum(entry.get("protein", 0) for entry in aggregates)
        total_carbs = sum(entry.get("carbs", 0) for entry in aggregates)
        total_fat = sum(entry.get("fat", 0) for entry in aggregates)
        print(f"\nTotals: {total_grams:.1f}g | {total_calories:.1f} kcal")
        print(f"Macros: P {total_protein:.1f}g | C {total_carbs:.1f}g | F {total_fat:.1f}g")

        # Save results
        writer.save_visuals(image_path=image_path, detections=results)
        writer.write_results(image_path=image_path, aggregates=aggregates, dish_first=dish_meta)
        writer.write_highest_predictions_csv(image_path=image_path, detections=results)

        # Validate against ground truth
        if ground_truth:
            validation_result = validate_against_ground_truth(
                detections=results,
                image_name=image_path.name,
                ground_truth=ground_truth,
                label_normalizer=label_normalizer,
            )
            validation_result["image_name"] = image_path.name
            validation_results.append(validation_result)

    # Save validation report
    if validation_results:
        writer.save_ground_truth_evaluation(validation_results)
        writer.copy_analysis_to_results()
        print_ground_truth_report(validation_results)
