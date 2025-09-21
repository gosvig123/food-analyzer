"""CLI entry point for running food inference over images in the data folder."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List
import json

from PIL import UnidentifiedImageError

from food_analyzer import FoodInferencePipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
RESULTS_DIR = Path("results")


def iter_image_paths(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
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


def write_results(image_path: Path, detections: List[dict], aggregates: List[dict], totals: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path),
        "top_detections": detections[:5],
        "detections": detections,
        "aggregated": aggregates,
        "totals": totals,
    }
    output_path = RESULTS_DIR / f"{image_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_inference(target_dir: Path) -> None:
    pipeline = FoodInferencePipeline()
    images = list(iter_image_paths(target_dir))

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

        write_results(
            image_path=image_path,
            detections=results,
            aggregates=aggregates,
            totals={"grams": total_grams, "calories": total_calories},
        )


def main(argv: list[str]) -> int:
    target = Path(argv[0]) if argv else Path("data")
    if not target.exists():
        print(f"Directory not found: {target}")
        return 1

    run_inference(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
