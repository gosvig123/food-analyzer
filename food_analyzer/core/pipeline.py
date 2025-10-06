"""High-level orchestration wiring detection, classification, and nutrition."""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, UnidentifiedImageError

from ..classification.classifier import FoodClassifier
from ..detection.depth import DepthEstimator
from ..detection.detector import FoodDetector
from .types import AnalyzedFood, ImageInput


class FoodInferencePipeline:
    """High-level orchestrator wiring detector → classifier → nutrition lookup."""

    def __init__(
        self,
        detector: FoodDetector | None = None,
        classifier: FoodClassifier | None = None,
        depth_estimator: DepthEstimator | None = None,
        use_detector_labels: bool = False,
        maximize_recall: bool = False,
    ) -> None:
        self.detector = detector or FoodDetector()
        self.classifier = classifier or FoodClassifier()
        self.depth_estimator = depth_estimator or DepthEstimator()
        self.use_detector_labels = use_detector_labels
        self.maximize_recall = bool(maximize_recall)

    def analyze(self, image_input: ImageInput) -> List[AnalyzedFood]:
        image = _load_image(image_input)
        detections = self.detector(image)

        depth_map = None
        depth_mean = None
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator(image)
            if depth_map is not None and depth_map.numel() > 0:
                depth_mean = float(depth_map.mean().item())

        analyzed: List[AnalyzedFood] = []
        for detection in detections:
            crop = image.crop(detection.box)
            # If we have a segmentation mask, use it to isolate and tightly crop the food area
            if getattr(detection, "mask_polygon", None):
                try:
                    mask_img = Image.new("L", image.size, 0)
                    draw = ImageDraw.Draw(mask_img)
                    poly_pts = [(int(x), int(y)) for x, y in detection.mask_polygon]
                    draw.polygon(poly_pts, fill=255)
                    # Compute tight bounding box around polygon with a small padding
                    xs = [x for x, _ in poly_pts]
                    ys = [y for _, y in poly_pts]
                    pad = 4
                    left = max(min(xs) - pad, 0)
                    top = max(min(ys) - pad, 0)
                    right = min(max(xs) + pad, image.size[0])
                    bottom = min(max(ys) + pad, image.size[1])
                    masked = Image.new("RGB", image.size, (255, 255, 255))
                    masked.paste(image, mask=mask_img)
                    crop = masked.crop((left, top, right, bottom))
                except Exception:
                    pass
            if self.use_detector_labels:
                classification = {
                    "label": detection.label,
                    "confidence": detection.confidence,
                }
            else:
                classification = self.classifier(crop)

            label = str(classification.get("label", detection.label))
            label_conf = float(classification.get("confidence", 1.0))
            candidates = classification.get("candidates") if isinstance(classification, dict) else None

            # Enhanced confidence combination with weighted scoring
            detection_weight = 0.3  # Detection confidence contributes 30%
            classification_weight = 0.7  # Classification confidence contributes 70%

            # Use harmonic mean for better balance when one confidence is very low
            if detection.confidence > 0 and label_conf > 0:
                combined_confidence = (
                    detection_weight * detection.confidence
                    + classification_weight * label_conf
                )
            else:
                combined_confidence = min(detection.confidence, label_conf)

            # Skip very low confidence detections to reduce noise unless maximizing recall
            min_combined_confidence = 0.0 if self.maximize_recall else 0.1
            if combined_confidence >= min_combined_confidence and label != "unknown":
                analyzed.append(
                    AnalyzedFood(
                        label=label,
                        confidence=float(combined_confidence),
                        box=detection.box,
                        mask_polygon=detection.mask_polygon,
                        candidates=candidates if isinstance(candidates, list) else None,
                    )
                )
        return analyzed


def analyze_image(image_input: ImageInput) -> List[AnalyzedFood]:
    """Functional helper mirroring :meth:`FoodInferencePipeline.analyze`."""

    pipeline = FoodInferencePipeline()
    return pipeline.analyze(image_input)


def _load_image(image_input: ImageInput) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input
    path = Path(image_input)
    if not path.exists():  # pragma: no cover - guard rail for manual use
        raise FileNotFoundError(path)
    try:
        return Image.open(path)
    except UnidentifiedImageError as exc:  # pragma: no cover
        raise UnidentifiedImageError(f"Unsupported image file: {path}") from exc


__all__ = ["FoodInferencePipeline", "analyze_image"]
