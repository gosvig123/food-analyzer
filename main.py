"""CLI entry point for running food inference over images with configurable settings."""

from __future__ import annotations

import sys
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Set
import json

from PIL import UnidentifiedImageError, Image, ImageDraw

from food_analyzer.classifier import FoodClassifier
from food_analyzer.config import load_config, resolve_path_relative_to_project
from food_analyzer.depth import DepthEstimator
from food_analyzer.detector import FoodDetector
from food_analyzer.nutrition import NutritionLookup
from food_analyzer.pipeline import FoodInferencePipeline
from food_analyzer.volume import VolumeEstimator


def iter_image_paths(directory: Path, extensions: Set[str]) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def format_result_row(index: int, result: dict) -> str:
    label = result["label"]
    confidence = float(result["confidence"])
    grams = float(result["portion_grams"])
    calories = float(result["nutrition"].get("calories", 0.0))
    return f"{index:>2} | {label:<20} | {confidence:>6.2f} | {grams:>7.1f}g | {calories:>7.1f} kcal"


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
        entry["grams"] += float(item["portion_grams"])
        entry["calories"] += float(item["nutrition"].get("calories", 0.0))
        entry["max_confidence"] = max(entry["max_confidence"], float(item["confidence"]))
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


def write_results(results_dir: Path, image_path: Path, detections: List[dict], aggregates: List[dict], totals: dict) -> None:
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



def save_visuals(results_dir: Path, image_path: Path, detections: List[dict], save_overlays: bool, save_crops: bool, save_masks: bool) -> None:
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
                    draw.polygon(poly_pts, fill=(255, 0, 0, 80), outline=(255, 0, 0, 180))
                except Exception:
                    pass
            # Draw bounding box + label
            if box and len(box) == 4:
                left, top, right, bottom = [int(v) for v in box]
                draw.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0, 220), width=3)
                txt = f"{label} {conf:.2f}"
                tw, th = ImageDraw.Draw(Image.new("RGB", (1,1))).textlength(txt), 12
                draw.rectangle([(left, max(0, top - th - 4)), (left + int(tw) + 6, top)], fill=(255, 0, 0, 220))
                draw.text((left + 3, max(0, top - th - 2)), txt, fill=(255, 255, 255, 255))
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
                    xs = [int(x) for x, _ in mask_poly]; ys = [int(y) for _, y in mask_poly]
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
            primary_crop.save(crops_dir / f"{image_path.stem}_{idx:02d}_{safe_label}.jpg")
            bbox_crop.save(crops_dir / "bbox" / f"{image_path.stem}_{idx:02d}_{safe_label}_bbox.jpg")
            (mask_crop if mask_crop is not None else bbox_crop).save(crops_dir / "mask" / f"{image_path.stem}_{idx:02d}_{safe_label}_mask.jpg")
            # Montage (side-by-side bbox vs mask-tight)
            try:
                left_img, right_img = bbox_crop, (mask_crop if mask_crop is not None else bbox_crop)
                h = max(left_img.height, right_img.height)
                new_left = Image.new("RGB", (left_img.width, h), (255, 255, 255))
                new_left.paste(left_img, (0, 0))
                new_right = Image.new("RGB", (right_img.width, h), (255, 255, 255))
                new_right.paste(right_img, (0, 0))
                montage = Image.new("RGB", (new_left.width + new_right.width, h), (255, 255, 255))
                montage.paste(new_left, (0, 0))
                montage.paste(new_right, (new_left.width, 0))
                (crops_dir / "montage").mkdir(parents=True, exist_ok=True)
                montage.save(crops_dir / "montage" / f"{image_path.stem}_{idx:02d}_{safe_label}_montage.jpg")
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
                mask_img.save(masks_dir / f"{image_path.stem}_{idx:02d}_{label}_mask.png")
            except Exception:
                continue


def build_pipeline_from_config(cfg: dict) -> FoodInferencePipeline:
    # Nutrition table
    table_path_str = cfg.get("nutrition", {}).get("table_path")
    table_path = resolve_path_relative_to_project(table_path_str) if table_path_str else None
    nutrition_lookup = NutritionLookup(table=None, table_path=table_path)

    # Classifier fallback labels derived from nutrition table keys
    fallback_labels = list(nutrition_lookup.table.keys()) if hasattr(nutrition_lookup, "table") else None

    # Components
    detector_cfg = cfg.get("detector", {})
    detector = FoodDetector(
        score_threshold=float(detector_cfg.get("score_threshold", 0.5)),
        iou_threshold=float(detector_cfg.get("iou_threshold", 0.45)),
        max_detections=int(detector_cfg.get("max_detections", 100)),
        device=detector_cfg.get("device"),
        backend=str(detector_cfg.get("backend", "torchvision_fasterrcnn")),
        model_name=detector_cfg.get("model_name") or None,
        imgsz=int(detector_cfg.get("imgsz")) if detector_cfg.get("imgsz") is not None else None,
        retina_masks=bool(detector_cfg.get("retina_masks", True)),
        augment=bool(detector_cfg.get("augment", False)),
        tta_imgsz=detector_cfg.get("tta_imgsz") or None,
        refine_masks=bool(detector_cfg.get("refine_masks", True)),
        refine_method=str(detector_cfg.get("refine_method", "morphology")),
        morph_kernel=int(detector_cfg.get("morph_kernel", 3)),
        morph_iters=int(detector_cfg.get("morph_iters", 1)),
        sam_checkpoint=detector_cfg.get("sam_checkpoint"),
        sam_model_type=detector_cfg.get("sam_model_type"),
        fusion_method=str(detector_cfg.get("fusion_method", "soft_nms")),
        soft_nms_sigma=float(detector_cfg.get("soft_nms_sigma", 0.5)),
    )

    classifier_cfg = cfg.get("classifier", {})
    labels_path = classifier_cfg.get("labels_path")
    resolved_labels_path = (
        resolve_path_relative_to_project(labels_path) if labels_path is not None else None
    )
    classifier = FoodClassifier(
        device=classifier_cfg.get("device"),
        fallback_labels=fallback_labels,
        backend=str(classifier_cfg.get("backend", "efficientnet_b0")),
        labels_path=resolved_labels_path,
    )

    volume_estimator = VolumeEstimator(
        grams_for_full_plate=float(cfg.get("volume", {}).get("grams_for_full_plate", 300.0))
    )

    depth_enabled = bool(cfg.get("depth", {}).get("enabled", True))
    depth_estimator = DepthEstimator() if depth_enabled else None

    pipeline_cfg = cfg.get("pipeline", {})
    return FoodInferencePipeline(
        detector=detector,
        classifier=classifier,
        volume_estimator=volume_estimator,
        nutrition_lookup=nutrition_lookup,
        depth_estimator=depth_estimator,
        use_detector_labels=bool(pipeline_cfg.get("use_detector_labels", False)),
    )


def run_inference(target_dir: Path, cfg: dict) -> None:
    pipeline = build_pipeline_from_config(cfg)

    exts = {e.lower() for e in cfg.get("io", {}).get("image_extensions", [".jpg", ".jpeg", ".png"]) }
    results_dir = Path(cfg.get("io", {}).get("results_dir", "results"))
    save_overlays = bool(cfg.get("io", {}).get("save_overlays", False))
    save_crops = bool(cfg.get("io", {}).get("save_crops", False))
    save_masks = bool(cfg.get("io", {}).get("save_masks", False))

    images = list(iter_image_paths(target_dir, exts))
    if not images:
        print(f"No images found in {target_dir}")
        return

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
    parser = argparse.ArgumentParser(description="Analyze food images and estimate nutrition.")
    parser.add_argument("directory", nargs="?", default="data", help="Directory containing images to analyze")
    parser.add_argument("--config", dest="config", default=None, help="Path to JSON/YAML config file")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation stage")
    parser.add_argument("--score-threshold", type=float, default=None, help="Detector score threshold override")
    parser.add_argument(
        "--grams-full-plate", type=float, default=None, help="Grams corresponding to full-frame portion"
    )
    parser.add_argument("--results-dir", default=None, help="Where to write JSON outputs (overrides config)")
    parser.add_argument("--compare-dirs", nargs="+", default=None, help="Compare detection counts across result folders")
    parser.add_argument("--compare-out", default=None, help="Optional CSV path for --compare-dirs output")
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    if args.no_depth:
        cfg.setdefault("depth", {})["enabled"] = False
    if args.score_threshold is not None:
        cfg.setdefault("detector", {})["score_threshold"] = float(args.score_threshold)
    if args.grams_full_plate is not None:
        cfg.setdefault("volume", {})["grams_for_full_plate"] = float(args.grams_full_plate)
    if args.results_dir is not None:
        cfg.setdefault("io", {})["results_dir"] = str(args.results_dir)
    return cfg


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.compare_dirs:
        compare_results(args.compare_dirs, out_path=args.compare_out)
        return 0
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
