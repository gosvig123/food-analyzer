"""Classification utilities to assign semantic labels to detected food items."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .ingredient_api import get_dynamic_ingredient_labels
from .intelligent_labels import get_intelligent_ingredient_labels


class FoodClassifier:
    """Image classifier with graceful degradation when weights are unavailable."""

    def __init__(
        self,
        device: str | None = None,
        backend: str = "efficientnet_b0",
        dynamic_labels_source: str | None = None,
        intelligent_labels_method: str | None = None,
        temperature: float = 1.0,
        confidence_threshold: float = 0.3,
        multi_scale: bool = False,
        ensemble_weights: list[float] | None = None,
    ) -> None:
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Any = None
        self.backend: str = backend
        self.labels: list[str] = []
        self.temperature: float = temperature
        self.confidence_threshold: float = confidence_threshold
        self.multi_scale: bool = multi_scale
        self.ensemble_weights: list[float] = ensemble_weights or [1.0, 1.0, 1.0]

        # Priority: intelligent labels (from model weights/embeddings) → API fallback
        if intelligent_labels_method:
            try:
                intelligent_labels = get_intelligent_ingredient_labels(
                    method=intelligent_labels_method
                )
                if intelligent_labels:
                    self.labels = intelligent_labels
            except Exception as exc:
                warnings.warn(
                    f"Failed to extract intelligent labels using {intelligent_labels_method}: {exc}"
                )

        if not self.labels and dynamic_labels_source:
            try:
                dynamic_labels = get_dynamic_ingredient_labels(
                    source=dynamic_labels_source
                )
                if dynamic_labels:
                    self.labels = dynamic_labels
            except Exception as exc:
                warnings.warn(
                    f"Failed to fetch dynamic labels from {dynamic_labels_source}: {exc}"
                )

        try:
            if backend == "clip_zeroshot":
                try:
                    import open_clip  # type: ignore
                except Exception as exc:
                    raise RuntimeError(f"open-clip-torch not available: {exc}")
                # Use ViT-L-14 with 336px resolution for maximum accuracy on food classification
                model_name = "ViT-L-14-336"
                pretrained = "openai"
                try:
                    self.clip_model: Any
                    self.preprocess: Any
                    self.clip_model, _, self.preprocess = (
                        open_clip.create_model_and_transforms(
                            model_name, pretrained=pretrained
                        )
                    )
                    self.clip_tokenizer: Any = open_clip.get_tokenizer(model_name)
                except Exception:
                    try:
                        # Try ViT-B-16 as intermediate fallback
                        self.clip_model, _, self.preprocess = (
                            open_clip.create_model_and_transforms(
                                "ViT-B-16", pretrained="laion2b_s34b_b88k"
                            )
                        )
                        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
                    except Exception:
                        # Final fallback to B-32 weights
                        self.clip_model, _, self.preprocess = (
                            open_clip.create_model_and_transforms(
                                "ViT-B-32", pretrained="laion2b_s34b_b79k"
                            )
                        )
                        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                _ = self.clip_model.to(self.device).eval()
                # Ensure labels exist
                if not self.labels:
                    raise ValueError(
                        "Classifier labels must be provided for ingredient-level zero-shot CLIP."
                    )
                # Enhanced prompt ensembling for better food/ingredient recognition
                templates = [
                    "a photo of {}",
                    "a high quality photo of {}",
                    "a close-up food photo of {}",
                    "a dish containing {}",
                    "ingredient: {}",
                    "fresh {}",
                    "cooked {}",
                    "raw {}",
                    "sliced {}",
                    "{} on a plate",
                    "organic {}",
                    "chopped {}",
                    "diced {}",
                    "grilled {}",
                    "baked {}",
                    "a serving of {}",
                    "natural {}",
                    "whole {}",
                    "food ingredient {}",
                    "{} ingredient",
                    "edible {}",
                    "{} in a bowl",
                    "{} on the table",
                    "prepared {}",
                    "red {}",
                    "green {}",
                    "ripe {}",
                    "{} vegetable",
                    "{} fruit",
                    "leafy {}",
                    "liquid {}",
                    "{} sauce",
                    # Additional food-specific templates for better recall
                    "a portion of {}",
                    "fresh {} ingredients",
                    "food with {}",
                    "meal containing {}",
                    "plate with {}",
                    "dish of {}",
                    "healthy {}",
                    "culinary ingredient {}",
                    "cooking with {}",
                    "nutrition from {}",
                    "food item: {}",
                    "edible {} food",
                    "dietary {}",
                    "kitchen ingredient {}",
                    "recipe ingredient {}",
                    "food component {}",
                    "visible {}",
                    "identifiable {}",
                    "cuisine ingredient {}",
                    "food preparation with {}",
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
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                self._clip_text_features: Any = text_features
            else:
                # Use EfficientNet-B4 for better accuracy than B0
                try:
                    import timm  # type: ignore
                    from timm.data import (  # type: ignore
                        create_transform,
                        resolve_data_config,
                    )
                except Exception as exc:
                    raise RuntimeError(f"timm not available: {exc}")
                self.model = timm.create_model("efficientnet_b4", pretrained=True)
                _ = self.model.to(self.device).eval()
                data_cfg = resolve_data_config({}, model=self.model)
                self.preprocess = create_transform(**data_cfg)
                if not self.labels:
                    num_classes = getattr(self.model, "num_classes", 1000)
                    self.labels = [f"class_{i}" for i in range(num_classes)]
        except Exception as exc:  # pragma: no cover - explicit failure (no heuristics)
            raise RuntimeError(f"Classifier backend initialization failed: {exc}")

        if not hasattr(self, "preprocess"):
            # Should only occur if model loading succeeded but preprocess not set
            self.preprocess = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )

        if not self.labels:
            # Ingredient-only policy: do not fabricate labels
            raise ValueError(
                "No classifier labels available; set classifier.labels_path to an ingredient list."
            )

    def __call__(self, image: Image.Image) -> dict[str, float | str]:
        image = image.convert("RGB")

        # CLIP zero-shot path (if available)
        if getattr(self, "clip_model", None) is not None and hasattr(
            self, "_clip_text_features"
        ):
            with torch.no_grad():
                # Multi-scale test-time augmentation for better recall
                if self.multi_scale:
                    # Use different crops and augmentations at the same final input size
                    augmentations = [
                        # Standard preprocessing
                        self.preprocess,
                        # Center crop - focuses on center of image
                        transforms.Compose(
                            [
                                transforms.CenterCrop(min(image.size)),
                                transforms.Resize(self.preprocess.transforms[0].size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        ),
                        # Random crop - captures different parts
                        transforms.Compose(
                            [
                                transforms.RandomCrop(min(image.size)),
                                transforms.Resize(self.preprocess.transforms[0].size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        ),
                        # Slight horizontal flip for different perspective
                        transforms.Compose(
                            [
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Resize(self.preprocess.transforms[0].size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.48145466, 0.4578275, 0.40821073),
                                    (0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        ),
                    ]

                    weights = self.ensemble_weights[: len(augmentations)]
                    all_logits = []

                    for i, aug_transform in enumerate(augmentations):
                        try:
                            aug_tensor = (
                                aug_transform(image).unsqueeze(0).to(self.device)
                            )
                            aug_features = self.clip_model.encode_image(aug_tensor)  # type: ignore
                            aug_features = aug_features / aug_features.norm(
                                dim=-1, keepdim=True
                            )
                            aug_logits = (
                                100.0 * aug_features @ self._clip_text_features.T
                            )  # type: ignore

                            # Apply ensemble weight
                            weight = weights[i] if i < len(weights) else 1.0
                            weighted_logits = aug_logits * weight
                            all_logits.append(weighted_logits)
                        except Exception:
                            # Fallback to original preprocessing if augmentation fails
                            orig_tensor = (
                                self.preprocess(image).unsqueeze(0).to(self.device)
                            )
                            orig_features = self.clip_model.encode_image(orig_tensor)  # type: ignore
                            orig_features = orig_features / orig_features.norm(
                                dim=-1, keepdim=True
                            )
                            orig_logits = (
                                100.0 * orig_features @ self._clip_text_features.T
                            )  # type: ignore
                            all_logits.append(orig_logits)

                    # Weighted ensemble of augmented predictions
                    if all_logits:
                        total_weight = sum(weights[: len(all_logits)])
                        logits = torch.stack(all_logits).sum(dim=0) / total_weight  # type: ignore
                    else:
                        # Fallback
                        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                        image_features = self.clip_model.encode_image(tensor)  # type: ignore
                        image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )
                        logits = 100.0 * image_features @ self._clip_text_features.T  # type: ignore
                else:
                    # Single inference using standard preprocessing
                    tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # type: ignore
                    image_features = self.clip_model.encode_image(tensor)  # type: ignore
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    logits = 100.0 * image_features @ self._clip_text_features.T  # type: ignore

                # Apply configurable temperature scaling for calibration
                calibrated_logits = logits / self.temperature  # type: ignore
                calibrated_probs = torch.nn.functional.softmax(
                    calibrated_logits,
                    dim=1,  # type: ignore
                )[0]

            if self.labels:
                conf, idx = calibrated_probs.max(dim=0)
                mapped_idx = int(idx.item())
                if 0 <= mapped_idx < len(self.labels):
                    # Additional confidence boost for ground truth ingredients
                    raw_conf = float(conf.item())
                    predicted_label = self.labels[mapped_idx]

                    # Apply confidence threshold
                    if raw_conf >= self.confidence_threshold:
                        return {
                            "label": predicted_label,
                            "confidence": float(raw_conf),
                        }
                    else:
                        # Below threshold - return low confidence result
                        return {"label": "unknown", "confidence": float(raw_conf)}

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

        # Fail gracefully for unknown items
        return {"label": "unknown", "confidence": 0.0}


__all__ = ["FoodClassifier"]
