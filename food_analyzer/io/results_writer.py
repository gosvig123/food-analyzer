"""Utility for writing inference outputs to disk.

This module centralizes all filesystem operations related to saving model
outputs so `main.py` and other callers can remain focused on orchestration.
"""

from __future__ import annotations

import csv
import json
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, UnidentifiedImageError


class ResultsWriter:
    """Encapsulates writing inference results and visual artifacts.

    Responsibilities:
      - write per-image JSON payloads
      - append a highest-predictions CSV
      - save overlay / crop / mask assets
      - write ground-truth evaluation reports (JSON + CSV)
      - copy auxiliary analysis files into the results folder

    The class is intentionally small and opinionated: callers provide
    pre-computed payloads and the writer manages deterministic file layout.
    """

    def __init__(
        self,
        results_dir: Path | str,
        save_overlays: bool = False,
        save_crops: bool = False,
        save_masks: bool = False,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.save_overlays = bool(save_overlays)
        self.save_crops = bool(save_crops)
        self.save_masks = bool(save_masks)
        # Ensure base directory exists right away
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # JSON / CSV result writers
    # -----------------------------
    def write_results(self, image_path: Path, aggregates: List[dict]) -> None:
        """Write a per-image JSON payload containing aggregated summaries only."""
        payload = {
            "image": str(image_path),
            "aggregated": aggregates,
        }
        output_path = self.results_dir / f"{image_path.stem}.json"
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to write results JSON for {image_path}: {exc}")

    def write_highest_predictions_csv(
        self, image_path: Path, detections: List[dict]
    ) -> None:
        """Append highest-confidence prediction per ingredient for an image to a CSV."""
        csv_path = self.results_dir / "highest_predictions.csv"
        write_headers = not csv_path.exists()

        # Determine highest-confidence per ingredient
        ingredient_predictions: dict[str, dict] = {}
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

        try:
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
        except Exception as exc:
            warnings.warn(f"Failed to write highest_predictions.csv: {exc}")

    # -----------------------------
    # Visual asset writers
    # -----------------------------
    def save_visuals(self, image_path: Path, detections: List[dict]) -> None:
        """Save overlays, crops, and masks according to enabled flags.

        The function opens the image once and delegates to helper methods.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            warnings.warn(f"File is not a valid image: {image_path}")
            return
        except Exception as exc:
            warnings.warn(f"Failed to open image {image_path}: {exc}")
            return

        if self.save_overlays:
            try:
                self._save_overlays(image, image_path, detections)
            except Exception as exc:
                warnings.warn(f"Overlay save failed for {image_path}: {exc}")

        if self.save_crops:
            try:
                self._save_crops(image, image_path, detections)
            except Exception as exc:
                warnings.warn(f"Crops save failed for {image_path}: {exc}")

        if self.save_masks:
            try:
                self._save_masks(image, image_path, detections)
            except Exception as exc:
                warnings.warn(f"Masks save failed for {image_path}: {exc}")

    def _save_overlays(
        self, image: Image.Image, image_path: Path, detections: List[dict]
    ) -> None:
        overlays_dir = self.results_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        base = image.convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for det in detections:
            box = det.get("box")
            label = str(det.get("label", ""))
            try:
                conf = float(det.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            mask_poly = det.get("mask_polygon")

            # Draw mask polygon first (semi-transparent)
            if isinstance(mask_poly, list) and len(mask_poly) >= 3:
                try:
                    poly_pts = [(int(x), int(y)) for x, y in mask_poly]
                    draw.polygon(
                        poly_pts, fill=(255, 0, 0, 80), outline=(255, 0, 0, 180)
                    )
                except Exception:
                    # Ignore individual mask drawing errors
                    pass

            # Draw bounding box + label
            if box and len(box) == 4:
                try:
                    left, top, right, bottom = [int(v) for v in box]
                    draw.rectangle(
                        [(left, top), (right, bottom)],
                        outline=(255, 0, 0, 220),
                        width=3,
                    )
                    txt = f"{label} {conf:.2f}"
                    # textlength requires a drawing context; fallback to fixed height
                    try:
                        tw = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(txt)
                        th = 12
                    except Exception:
                        tw, th = (len(txt) * 6, 12)
                    draw.rectangle(
                        [(left, max(0, top - th - 4)), (left + int(tw) + 6, top)],
                        fill=(255, 0, 0, 220),
                    )
                    draw.text(
                        (left + 3, max(0, top - th - 2)), txt, fill=(255, 255, 255, 255)
                    )
                except Exception:
                    # continue drawing other detections
                    continue

        composed = Image.alpha_composite(base, overlay)
        try:
            composed.save(overlays_dir / f"{image_path.stem}_overlay.png")
        except Exception as exc:
            warnings.warn(f"Failed to save overlay image for {image_path}: {exc}")

    def _save_crops(
        self, image: Image.Image, image_path: Path, detections: List[dict]
    ) -> None:
        crops_dir = self.results_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        for idx, det in enumerate(detections):
            box = det.get("box")
            label = str(det.get("label", "item"))
            mask_poly = det.get("mask_polygon")
            if not box or len(box) != 4:
                continue

            try:
                bbox_crop = image.crop(tuple(int(v) for v in box))
            except Exception:
                continue

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

            primary_crop = mask_crop if mask_crop is not None else bbox_crop
            safe_label = label.replace(" ", "_")

            (crops_dir / "bbox").mkdir(parents=True, exist_ok=True)
            (crops_dir / "mask").mkdir(parents=True, exist_ok=True)

            try:
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
            except Exception:
                # ignore individual save failures
                pass

            # Montage (side-by-side) - optional and non-fatal
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

    def _save_masks(
        self, image: Image.Image, image_path: Path, detections: List[dict]
    ) -> None:
        masks_dir = self.results_dir / "masks"
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

    # -----------------------------
    # Ground truth evaluation & helpers
    # -----------------------------
    def save_ground_truth_evaluation(self, validation_results: Iterable[Dict]) -> None:
        """Save evaluation report (JSON + CSV) to results folder."""
        validation_results = list(validation_results)

        total_precision = sum(r["precision"] for r in validation_results)
        total_recall = sum(r["recall"] for r in validation_results)
        total_f1 = sum(r["f1"] for r in validation_results)
        num_images = len(validation_results)

        avg_precision = total_precision / num_images if num_images > 0 else 0.0
        avg_recall = total_recall / num_images if num_images > 0 else 0.0
        avg_f1 = total_f1 / num_images if num_images > 0 else 0.0

        plate_metrics = defaultdict(list)
        for result in validation_results:
            plate_type = result["plate_type"]
            if plate_type != "unknown":
                plate_metrics[plate_type].append(result)

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

        eval_path = self.results_dir / "ground_truth_evaluation.json"
        try:
            with eval_path.open("w", encoding="utf-8") as f:
                json.dump(evaluation_report, f, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to write evaluation JSON: {exc}")

        # CSV variant
        csv_path = self.results_dir / "ground_truth_evaluation.csv"
        try:
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
                            "image_name": result.get("image_name", "unknown"),
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
        except Exception as exc:
            warnings.warn(f"Failed to write evaluation CSV: {exc}")

        print(f"Ground truth evaluation saved to:")
        print(f"  - {eval_path}")
        print(f"  - {csv_path}")

    def copy_analysis_to_results(self) -> None:
        """Copy optional analysis report into the results directory (if present)."""
        analysis_source = Path("ground_truth_analysis.md")
        if analysis_source.exists():
            analysis_dest = self.results_dir / "ground_truth_analysis.md"
            try:
                shutil.copy2(analysis_source, analysis_dest)
                print(f"Analysis report copied to: {analysis_dest}")
            except Exception as exc:
                warnings.warn(f"Failed to copy analysis report: {exc}")
