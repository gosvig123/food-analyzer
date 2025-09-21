"""Classification utilities to assign semantic labels to detected food items."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence
import json
import warnings

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet as efficientnet_models




def _load_labels_from_path(path: str | Path | None) -> List[str]:
    if path is None:
        return []
    label_path = Path(path)
    if not label_path.exists():
        warnings.warn(f"Classifier labels file not found: {label_path}")
        return []
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        warnings.warn(f"Failed to load classifier labels from {label_path}: {exc}")
        return []
    if isinstance(payload, Sequence) and all(isinstance(item, str) for item in payload):
        return list(payload)
    warnings.warn(f"Classifier labels file must be a list of strings: {label_path}")
    return []


class FoodClassifier:
    """Image classifier with graceful degradation when weights are unavailable."""

    def __init__(
        self,
        device: str | None = None,
        fallback_labels: Iterable[str] | None = None,
        backend: str = "efficientnet_b0",
        labels_path: str | Path | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.backend = backend
        self.labels: List[str] = []

        user_labels = _load_labels_from_path(labels_path)
        if user_labels:
            self.labels = user_labels
        elif fallback_labels:
            self.labels = list(fallback_labels)

        try:
            if backend == "efficientnet_b4":
                try:
                    import timm  # type: ignore
                    from timm.data import resolve_data_config, create_transform  # type: ignore
                except Exception as exc:
                    raise RuntimeError(f"timm not available: {exc}")
                self.model = timm.create_model("efficientnet_b4", pretrained=True)
                self.model.to(self.device).eval()
                data_cfg = resolve_data_config({}, model=self.model)
                self.preprocess = create_transform(**data_cfg)
                if not self.labels:
                    num_classes = getattr(self.model, "num_classes", 1000)
                    self.labels = [f"class_{i}" for i in range(num_classes)]
            elif backend == "clip_zeroshot":
                try:
                    import open_clip  # type: ignore
                except Exception as exc:
                    raise RuntimeError(f"open-clip-torch not available: {exc}")
                # Prefer a stronger CLIP backbone for better zero-shot performance
                model_name = "ViT-L-14"
                pretrained = "laion2b_s32b_b82k"
                try:
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
                    self.clip_tokenizer = open_clip.get_tokenizer(model_name)
                except Exception:
                    # Fallback to B-32 weights if L-14 unavailable
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
                    self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self.clip_model.to(self.device).eval()
                # Ensure labels exist
                if not self.labels:
                    raise ValueError("Classifier labels must be provided for ingredient-level zero-shot CLIP.")
                # Prompt ensembling improves ingredient recognition
                templates = [
                    "a photo of {}",
                    "a close-up food photo of {}",
                    "a dish with {}",
                    "ingredient: {}",
                    "raw {}",
                    "chopped {}",
                    "sliced {}",
                ]
                with torch.no_grad():
                    feats = []
                    for t in templates:
                        prompts = [t.format(lbl) for lbl in self.labels]
                        text = self.clip_tokenizer(prompts)
                        f = self.clip_model.encode_text(text.to(self.device))
                        f = f / f.norm(dim=-1, keepdim=True)
                        feats.append(f)
                    text_features = sum(feats) / len(feats)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self._clip_text_features = text_features
            else:
                weights = efficientnet_models.EfficientNet_B0_Weights.DEFAULT
                self.model = efficientnet_models.efficientnet_b0(weights=weights)
                self.model.to(self.device).eval()
                self.preprocess = weights.transforms()
                categories: Iterable[str] | None = weights.meta.get("categories")
                if categories and not self.labels:
                    self.labels = list(categories)
        except Exception as exc:  # pragma: no cover - explicit failure (no heuristics)
            raise RuntimeError(f"Classifier backend initialization failed: {exc}")

        if not hasattr(self, "preprocess"):
            # Should only occur if model loading succeeded but preprocess not set
            self.preprocess = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )

        if not self.labels:
            # Ingredient-only policy: do not fabricate labels
            raise ValueError("No classifier labels available; set classifier.labels_path to an ingredient list.")

    def __call__(self, image: Image.Image) -> dict[str, float | str]:
        image = image.convert("RGB")
        tensor = self.preprocess(image)

        # CLIP zero-shot path (if available)
        if getattr(self, "clip_model", None) is not None and hasattr(self, "_clip_text_features"):
            with torch.no_grad():
                image_features = self.clip_model.encode_image(tensor.unsqueeze(0).to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ self._clip_text_features.T
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            if self.labels:
                conf, idx = probs.max(dim=0)
                mapped_idx = int(idx.item())
                if 0 <= mapped_idx < len(self.labels):
                    return {"label": self.labels[mapped_idx], "confidence": float(conf.item())}

        if self.model is not None:
            with torch.no_grad():
                logits = self.model(tensor.unsqueeze(0).to(self.device))
                probs = torch.nn.functional.softmax(logits, dim=1)[0]

            if self.labels:
                conf, idx = probs.max(dim=0)
                mapped_idx = idx.item()
                if len(self.labels) < probs.shape[0]:
                    mapped_idx = mapped_idx % len(self.labels)
                if 0 <= mapped_idx < len(self.labels):
                    label = self.labels[mapped_idx]
                    return {"label": label, "confidence": float(conf.item())}

        # Ingredient-only policy: no heuristic predictions
        return {"label": "unknown", "confidence": 0.0}

    def _heuristic_label(self, tensor: torch.Tensor) -> tuple[str, float]:
        mean_channels = tensor.mean(dim=[1, 2])
        red, green, blue = (float(val) for val in mean_channels)
        brightness = float(tensor.mean().item())
        saturation = float((tensor.max() - tensor.min()).item())
        warmth = red - blue
        greenery = green - red
        yellowish = min(red, green) - blue

        if greenery > 0.05 and green > 0.3:
            label = self._prefer_label(["salad", "broccoli", "spinach", "lettuce", "cucumber"])
            confidence = min(0.9, (green + greenery) / 2)
        elif yellowish > 0.05 and brightness > 0.4:
            label = self._prefer_label(["pizza", "pasta", "bread", "wrap", "sandwich"])
            confidence = min(0.85, (red + green) / 2)
        elif warmth > 0.1 and red > 0.35:
            label = self._prefer_label(["burger", "steak", "chicken", "taco", "burrito"])
            confidence = min(0.8, red)
        elif blue > 0.38:
            label = self._prefer_label(["smoothie", "blueberry", "ice_cream", "juice", "tea"])
            confidence = min(0.75, blue)
        elif brightness > 0.65:
            label = self._prefer_label(["rice", "yogurt", "pancake", "cake", "noodles"])
            confidence = min(0.7, brightness)
        elif saturation < 0.25:
            label = self._prefer_label(["soup", "noodles", "dumpling", "sushi", "rice"])
            confidence = min(0.6, max(red, green, blue))
        else:
            label = self._prefer_label(["fish", "shrimp", "egg", "wrap", "sandwich"])
            confidence = min(0.65, max(red, green, blue))

        return label, float(max(0.05, min(confidence, 1.0)))

    def _prefer_label(self, candidates: Iterable[str]) -> str:
        for candidate in candidates:
            if candidate in self.labels:
                return candidate
        return self.labels[0] if self.labels else "generic_food"


__all__ = ["FoodClassifier"]
