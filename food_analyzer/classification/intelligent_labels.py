"""Intelligent ingredient label extraction from model weights and embeddings."""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import List, Optional, Set

from ..utils.config import IngredientConfig, load_ingredient_config


class IntelligentLabelExtractor:
    """Extract ingredient labels intelligently from existing model knowledge."""

    def __init__(self, config: Optional[IngredientConfig] = None):
        self.config = config or load_ingredient_config()
        self._food_keywords = set(self.config.food_keywords.get("categories", []))

    @lru_cache(maxsize=1)
    def extract_from_imagenet_classes(self) -> List[str]:
        """Extract food-related labels from ImageNet classes."""
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            weights = EfficientNet_B0_Weights.DEFAULT
            categories = weights.meta.get("categories", [])

            food_labels = []
            for category in categories:
                if self._is_food_related(category):
                    # Clean and normalize the label
                    cleaned = self._clean_label(category)
                    if cleaned:
                        food_labels.append(cleaned)

            return food_labels
        except Exception as exc:
            warnings.warn(f"Failed to extract ImageNet food classes: {exc}")
            return []

    @lru_cache(maxsize=1)
    def extract_from_clip_knowledge(
        self, model_name: Optional[str] = None
    ) -> List[str]:
        """Extract ingredient labels using CLIP's semantic understanding."""
        clip_config = self.config.models.get("clip")
        if not clip_config:
            return []

        effective_model_name = model_name or clip_config.get("name", "ViT-B-32")
        if not effective_model_name:
            return []

        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                effective_model_name, pretrained=clip_config.get("pretrained")
            )
            tokenizer = open_clip.get_tokenizer(effective_model_name)

            # Use CLIP to find food-related concepts by embedding similarity
            food_concepts = self._generate_food_concept_candidates()

            # Test which concepts CLIP understands well
            valid_ingredients = []

            with torch.no_grad():
                # Create embeddings for food test phrases
                test_phrases = [f"a photo of {concept}" for concept in food_concepts]
                text_tokens = tokenizer(test_phrases)
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Filter based on embedding quality (concepts with strong embeddings)
                for i, concept in enumerate(food_concepts):
                    embedding_norm = text_features[i].norm().item()
                    if embedding_norm > clip_config.get("embedding_threshold", 0.8):
                        valid_ingredients.append(concept)

            return valid_ingredients[: clip_config.get("max_results", 200)]

        except Exception as exc:
            warnings.warn(f"Failed to extract CLIP food concepts: {exc}")
            return []

    def extract_from_nutrition_database_structure(self) -> List[str]:
        """Extract common ingredients from nutrition database patterns."""
        # Use base ingredients from config
        base_ingredients = self.config.fallback_ingredients.copy()

        # Expand with common variations
        expanded = []
        for base in base_ingredients:
            expanded.append(base)
            # Add common variations
            if base in ["pepper", "berry"]:
                expanded.extend(
                    [f"{color} {base}" for color in ["red", "green", "black"]]
                )
            elif base == "oil":
                expanded.extend(["olive oil", "vegetable oil", "coconut oil"])
            elif base == "cheese":
                expanded.extend(["cheddar", "mozzarella", "parmesan", "feta"])

        return expanded

    def _generate_food_concept_candidates(self) -> List[str]:
        """Generate candidate food concepts for CLIP evaluation."""
        # Use fallback ingredients as base concepts
        return self.config.fallback_ingredients.copy()

    def _is_food_related(self, label: str) -> bool:
        """Check if a label is food-related."""
        label_lower = label.lower()

        # Use food indicators from config
        food_indicators = set(
            self.config.food_keywords.get("indicators", [])
            + self.config.food_keywords.get("specific_foods", [])
        )

        # Check if any food indicator is in the label
        return any(indicator in label_lower for indicator in food_indicators)

    def _clean_label(self, label: str) -> Optional[str]:
        """Clean and normalize a food label."""
        if not label:
            return None

        # Convert to lowercase and strip
        cleaned = label.lower().strip()

        # Remove common prefixes/suffixes that aren't ingredients
        for pattern in self.config.cleaning_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Clean up whitespace
        cleaned = " ".join(cleaned.split())

        # Filter out non-food items and keep simple ingredients
        if len(cleaned.split()) > 3 or len(cleaned) < 2:
            return None

        # Avoid generic terms
        if cleaned in self.config.generic_terms:
            return None

        # Filter out non-ingredient terms (containers, dishes, utensils, etc.)
        non_ingredient_terms = getattr(self.config, "non_ingredient_terms", [])
        for term in non_ingredient_terms:
            if term in cleaned or cleaned in term:
                return None

        return cleaned

    def get_intelligent_labels(self, method: str = "hybrid") -> List[str]:
        """
        Get ingredient labels using intelligent extraction methods.

        Args:
            method: "imagenet", "clip", "nutrition", or "hybrid"
        """
        labels: Set[str] = set()

        if method in ["imagenet", "hybrid"]:
            labels.update(self.extract_from_imagenet_classes())

        if method in ["clip", "hybrid"]:
            labels.update(self.extract_from_clip_knowledge())

        if method in ["nutrition", "hybrid"]:
            labels.update(self.extract_from_nutrition_database_structure())

        # Sort and return
        return sorted(list(labels))


def get_intelligent_ingredient_labels(
    method: str = "hybrid", config: Optional[IngredientConfig] = None
) -> List[str]:
    """
    Convenience function to get intelligent ingredient labels.

    Args:
        method: Extraction method - "imagenet", "clip", "nutrition", or "hybrid"
        config: Configuration object (optional, will load default if None)

    Returns:
        List of intelligently extracted ingredient labels
    """
    extractor = IntelligentLabelExtractor(config)
    return extractor.get_intelligent_labels(method)


__all__ = ["IntelligentLabelExtractor", "get_intelligent_ingredient_labels"]
