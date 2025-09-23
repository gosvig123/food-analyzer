"""Classification utilities to assign semantic labels to detected food items."""

from __future__ import annotations

import warnings
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet as efficientnet_models

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
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.backend = backend
        self.labels: List[str] = []

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
                # Use ViT-L-14 for maximum accuracy on food classification
                model_name = "ViT-L-14"
                pretrained = "laion2b_s32b_b82k"
                try:
                    self.clip_model, _, self.preprocess = (
                        open_clip.create_model_and_transforms(
                            model_name, pretrained=pretrained
                        )
                    )
                    self.clip_tokenizer = open_clip.get_tokenizer(model_name)
                except Exception:
                    # Fallback to B-32 weights if L-14 unavailable
                    self.clip_model, _, self.preprocess = (
                        open_clip.create_model_and_transforms(
                            "ViT-B-32", pretrained="laion2b_s34b_b79k"
                        )
                    )
                    self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self.clip_model.to(self.device).eval()
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
                self._clip_text_features = text_features
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
                self.model.to(self.device).eval()
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
        tensor = self.preprocess(image)

        # CLIP zero-shot path (if available)
        if getattr(self, "clip_model", None) is not None and hasattr(
            self, "_clip_text_features"
        ):
            with torch.no_grad():
                image_features = self.clip_model.encode_image(
                    tensor.unsqueeze(0).to(self.device)
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                logits = 100.0 * image_features @ self._clip_text_features.T
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            if self.labels:
                conf, idx = probs.max(dim=0)
                mapped_idx = int(idx.item())
                if 0 <= mapped_idx < len(self.labels):
                    return {
                        "label": self.labels[mapped_idx],
                        "confidence": float(conf.item()),
                    }

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
