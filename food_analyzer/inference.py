"""Minimal inference pipeline for food recognition and nutrition estimation.

The implementation follows the project plan at a reduced scope so we can run
basic end-to-end predictions without a fully trained model. Each stage attempts
to load a sensible default model and falls back to lightweight heuristics when
pre-trained weights are unavailable (e.g. offline environments)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
import warnings

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import detection as detection_models
from torchvision.models import efficientnet as efficientnet_models


ImageInput = Union[str, Path, Image.Image]


@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    confidence: float
    label: str


@dataclass
class AnalyzedFood:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]
    portion_grams: float
    nutrition: Dict[str, float]


class FoodDetector:
    """Wrapper around a detection model with a heuristic fallback."""

    def __init__(self, score_threshold: float = 0.5, device: str | None = None) -> None:
        self.score_threshold = score_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.categories: List[str] = ["food"]
        self.preprocess = transforms.Compose([transforms.ToTensor()])

        try:
            weights = detection_models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = detection_models.fasterrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device).eval()
            self.categories = weights.meta.get("categories", self.categories)
        except Exception as exc:  # pragma: no cover - best-effort offline fallback
            warnings.warn(f"Falling back to heuristic detector: {exc}")
            self.model = None

    def __call__(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")
        tensor = self.preprocess(image)

        if self.model is not None:
            with torch.no_grad():
                outputs = self.model([tensor.to(self.device)])[0]

            detections: List[Detection] = []
            for score, label_idx, box in zip(
                outputs["scores"].tolist(),
                outputs["labels"].tolist(),
                outputs["boxes"].tolist(),
            ):
                if score < self.score_threshold:
                    continue
                left, top, right, bottom = (int(round(v)) for v in box)
                category = "food"
                if 0 <= label_idx < len(self.categories):
                    category = self.categories[label_idx]
                detections.append(
                    Detection(
                        box=(left, top, right, bottom),
                        confidence=float(score),
                        label=category,
                    )
                )

            if detections:
                return detections

        # Fallback: treat the entire image as a single food item.
        width, height = image.size
        return [
            Detection(
                box=(0, 0, width, height),
                confidence=1.0,
                label="food",
            )
        ]


class FoodClassifier:
    """Image classifier with graceful degradation when weights are unavailable."""

    _fallback_labels = ["generic_food", "salad", "soup", "dessert", "drink"]

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labels = self._fallback_labels

        try:
            weights = efficientnet_models.EfficientNet_B0_Weights.DEFAULT
            self.model = efficientnet_models.efficientnet_b0(weights=weights)
            self.model.to(self.device).eval()
            self.preprocess = weights.transforms()
            categories: Iterable[str] | None = weights.meta.get("categories")
            if categories:
                self.labels = list(categories)
        except Exception as exc:  # pragma: no cover - best-effort offline fallback
            warnings.warn(f"Falling back to heuristic classifier: {exc}")
            self.preprocess = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )

    def __call__(self, image: Image.Image) -> Dict[str, float | str]:
        image = image.convert("RGB")
        tensor = self.preprocess(image)

        if self.model is not None:
            with torch.no_grad():
                logits = self.model(tensor.unsqueeze(0).to(self.device))
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            conf, idx = probs.max(dim=0)
            label = self.labels[idx.item()] if idx.item() < len(self.labels) else "generic_food"
            return {"label": label, "confidence": float(conf.item())}

        # Heuristic: classify based on dominant colour channel.
        mean_channels = tensor.mean(dim=[1, 2])
        red, green, blue = (float(val) for val in mean_channels)
        label = "generic_food"
        confidence = max(red, green, blue)
        if green > red and green > blue:
            label = "salad"
        elif red > blue and red > green:
            label = "soup"
        elif blue > 0.4:
            label = "drink"
        return {"label": label, "confidence": float(min(confidence, 1.0))}


class DepthEstimator:
    """Predicts per-pixel relative depth using a MiDaS backbone."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self.model.to(self.device).eval()
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self.transform = transforms.small_transform
        except Exception as exc:  # pragma: no cover - best-effort offline fallback
            warnings.warn(f"Depth estimator unavailable, using flat depth map: {exc}")
            self.model = None
            self.transform = None

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.model is None or self.transform is None:
            width, height = image.size
            return torch.ones((height, width))

        array = np.asarray(image)
        input_batch = self.transform(array).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(0)
            prediction = F.interpolate(
                prediction,
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
        return prediction.squeeze().cpu()


class VolumeEstimator:
    """Estimates food mass from bounding-box area relative to the frame."""

    def __init__(self, grams_for_full_plate: float = 300.0) -> None:
        self.grams_for_full_plate = grams_for_full_plate

    def __call__(
        self,
        box: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        depth_map: torch.Tensor | None = None,
        depth_mean: float | None = None,
    ) -> float:
        left, top, right, bottom = box
        width, height = image_size
        frame_area = max(width * height, 1)
        box_area = max((right - left) * (bottom - top), 1)
        ratio = min(box_area / frame_area, 1.0)

        depth_scale = 1.0
        if depth_map is not None:
            h, w = depth_map.shape[-2:]
            left_i = max(int(left), 0)
            right_i = min(int(right), w)
            top_i = max(int(top), 0)
            bottom_i = min(int(bottom), h)
            region = depth_map[top_i:bottom_i, left_i:right_i]
            if region.numel() > 0:
                global_mean = depth_mean or float(depth_map.mean().item())
                if global_mean > 0:
                    region_mean = float(region.mean().item())
                    depth_scale = max(0.5, min(region_mean / global_mean, 2.0))

        return round(self.grams_for_full_plate * ratio * depth_scale, 1)


class NutritionLookup:
    """Looks up macro nutrients using a tiny built-in table."""

    def __init__(self, table: Dict[str, Dict[str, float]] | None = None) -> None:
        self.table = table or {
            "generic_food": {"calories": 200.0, "protein": 8.0, "carbs": 25.0, "fat": 8.0},
            "salad": {"calories": 90.0, "protein": 4.0, "carbs": 12.0, "fat": 3.0},
            "soup": {"calories": 120.0, "protein": 6.0, "carbs": 14.0, "fat": 4.0},
            "dessert": {"calories": 250.0, "protein": 5.0, "carbs": 30.0, "fat": 12.0},
            "drink": {"calories": 60.0, "protein": 1.0, "carbs": 15.0, "fat": 0.0},
        }
        self.default_label = next(iter(self.table.keys()))

    def __call__(self, label: str, grams: float) -> Dict[str, float]:
        record = self.table.get(label, self.table[self.default_label])
        multiplier = grams / 100.0
        return {key: round(value * multiplier, 2) for key, value in record.items()}


class FoodInferencePipeline:
    """High-level orchestrator wiring detector → classifier → nutrition lookup."""

    def __init__(
        self,
        detector: FoodDetector | None = None,
        classifier: FoodClassifier | None = None,
        volume_estimator: VolumeEstimator | None = None,
        nutrition_lookup: NutritionLookup | None = None,
        depth_estimator: "DepthEstimator" | None = None,
    ) -> None:
        self.detector = detector or FoodDetector()
        self.classifier = classifier or FoodClassifier()
        self.volume_estimator = volume_estimator or VolumeEstimator()
        self.nutrition_lookup = nutrition_lookup or NutritionLookup()
        self.depth_estimator = depth_estimator or DepthEstimator()

    def analyze(self, image_input: ImageInput) -> List[AnalyzedFood]:
        image = self._load_image(image_input)
        detections = self.detector(image)
        depth_map = None
        depth_mean = None
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator(image)
            if depth_map is not None:
                depth_mean = float(depth_map.mean().item()) if depth_map.numel() > 0 else None

        analyzed: List[AnalyzedFood] = []
        for detection in detections:
            crop = image.crop(detection.box)
            classification = self.classifier(crop)
            grams = self.volume_estimator(
                detection.box,
                image.size,
                depth_map=depth_map,
                depth_mean=depth_mean,
            )
            nutrition = self.nutrition_lookup(classification["label"], grams)
            analyzed.append(
                AnalyzedFood(
                    label=str(classification["label"]),
                    confidence=float(
                        min(detection.confidence, float(classification.get("confidence", 1.0)))
                    ),
                    box=detection.box,
                    portion_grams=grams,
                    nutrition=nutrition,
                )
            )
        return analyzed

    @staticmethod
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


def analyze_image(image_input: ImageInput) -> List[AnalyzedFood]:
    """Functional helper mirroring :meth:`FoodInferencePipeline.analyze`."""

    pipeline = FoodInferencePipeline()
    return pipeline.analyze(image_input)


__all__ = [
    "AnalyzedFood",
    "DepthEstimator",
    "FoodInferencePipeline",
    "analyze_image",
]
