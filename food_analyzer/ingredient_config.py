"""Configuration management for ingredient label fetching."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class APIConfig:
    """Configuration for API endpoints."""

    base_url: str
    timeout: int = 10
    page_size: int = 50
    max_results: int = 500
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    name: str
    pretrained: Optional[str] = None
    embedding_threshold: float = 0.8
    max_results: int = 200


@dataclass
class IngredientConfig:
    """Main configuration for ingredient label fetching."""

    # API configurations
    apis: Dict[str, APIConfig] = field(default_factory=dict)

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Search categories for APIs
    search_categories: List[str] = field(default_factory=list)

    # Food classification keywords
    food_keywords: Dict[str, List[str]] = field(default_factory=dict)

    # Text cleaning patterns
    cleaning_patterns: List[str] = field(default_factory=list)

    # Generic terms to filter out
    generic_terms: List[str] = field(default_factory=list)

    # Cache settings
    cache_maxsize: int = 1
    cache_file: str = "ingredient_cache.json"

    # Fallback ingredients
    fallback_ingredients: List[str] = field(default_factory=list)

    @classmethod
    def load_from_file(cls, config_path: str | Path) -> "IngredientConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_path)

        if not config_path.exists():
            return cls.get_default_config()

        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as exc:
            warnings.warn(f"Failed to load config from {config_path}: {exc}")
            return cls.get_default_config()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngredientConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Load API configurations
        if "apis" in data:
            for name, api_data in data["apis"].items():
                config.apis[name] = APIConfig(**api_data)

        # Load model configurations
        if "models" in data:
            for name, model_data in data["models"].items():
                config.models[name] = ModelConfig(**model_data)

        # Load simple fields
        simple_fields = [
            "search_categories",
            "food_keywords",
            "cleaning_patterns",
            "generic_terms",
            "cache_maxsize",
            "cache_file",
            "fallback_ingredients",
        ]

        for field_name in simple_fields:
            if field_name in data:
                setattr(config, field_name, data[field_name])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "apis": {
                name: {
                    "base_url": api.base_url,
                    "timeout": api.timeout,
                    "page_size": api.page_size,
                    "max_results": api.max_results,
                    "headers": api.headers,
                    "params": api.params,
                }
                for name, api in self.apis.items()
            },
            "models": {
                name: {
                    "name": model.name,
                    "pretrained": model.pretrained,
                    "embedding_threshold": model.embedding_threshold,
                    "max_results": model.max_results,
                }
                for name, model in self.models.items()
            },
            "search_categories": self.search_categories,
            "food_keywords": self.food_keywords,
            "cleaning_patterns": self.cleaning_patterns,
            "generic_terms": self.generic_terms,
            "cache_maxsize": self.cache_maxsize,
            "cache_file": self.cache_file,
            "fallback_ingredients": self.fallback_ingredients,
        }

    def save_to_file(self, config_path: str | Path) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to save config to {config_path}: {exc}")

    @classmethod
    def get_default_config(cls) -> "IngredientConfig":
        """Get default configuration with sensible defaults."""
        config = cls()

        # Default API configurations
        config.apis = {
            "usda": APIConfig(
                base_url="https://api.nal.usda.gov/fdc/v1/foods/search",
                timeout=10,
                page_size=50,
                max_results=500,
                headers={"Accept": "application/json"},
                params={
                    "dataType": ["Foundation", "SR Legacy"],
                    "sortBy": "publishedDate",
                    "sortOrder": "desc",
                },
            ),
            "openfoodfacts": APIConfig(
                base_url="https://world.openfoodfacts.org/cgi/ingredients.pl",
                timeout=15,
                page_size=100,
                max_results=500,
                headers={"User-Agent": "FoodAnalyzer/1.0"},
                params={"json": "1"},
            ),
        }

        # Default model configurations
        config.models = {
            "clip": ModelConfig(
                name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                embedding_threshold=0.8,
                max_results=200,
            ),
            "efficientnet": ModelConfig(name="efficientnet_b0", max_results=300),
        }

        # Default search categories
        config.search_categories = [
            "fruits",
            "vegetables",
            "grains",
            "dairy",
            "meat",
            "seafood",
            "nuts",
            "spices",
            "herbs",
            "beans",
            "oils",
            "proteins",
        ]

        # Default food keywords by category
        config.food_keywords = {
            "categories": [
                "fruits",
                "vegetables",
                "meat",
                "dairy",
                "grains",
                "seafood",
                "herbs",
                "spices",
                "nuts",
                "seeds",
                "beans",
                "legumes",
                "oils",
            ],
            "indicators": [
                "food",
                "fruit",
                "vegetable",
                "meat",
                "fish",
                "dairy",
                "cheese",
                "bread",
                "soup",
                "salad",
                "sandwich",
                "pizza",
                "pasta",
                "rice",
                "noodle",
                "egg",
                "milk",
                "butter",
                "oil",
                "sauce",
                "spice",
                "herb",
            ],
            "specific_foods": [
                "apple",
                "orange",
                "banana",
                "tomato",
                "potato",
                "onion",
                "chicken",
                "beef",
                "pork",
                "salmon",
                "tuna",
            ],
        }

        # Default cleaning patterns
        config.cleaning_patterns = [
            r"\b(raw|cooked|fresh|frozen|canned|dried|organic)\b",
            r"\b(whole|half|quarter|sliced|diced|chopped)\b",
            r"\b(large|small|medium|mini)\b",
            r"\([^)]*\)",  # Remove parenthetical content
            r"\d+\s*(oz|lb|kg|g|ml|l)\b",  # Remove measurements
        ]

        # Default generic terms to filter out
        config.generic_terms = [
            "food",
            "item",
            "product",
            "dish",
            "meal",
            "snack",
            "beverage",
        ]

        # Default fallback ingredients
        config.fallback_ingredients = [
            "apple",
            "banana",
            "orange",
            "lemon",
            "lime",
            "strawberry",
            "blueberry",
            "tomato",
            "cucumber",
            "onion",
            "garlic",
            "carrot",
            "broccoli",
            "spinach",
            "lettuce",
            "bell pepper",
            "mushroom",
            "celery",
            "avocado",
            "corn",
            "peas",
            "chicken",
            "beef",
            "pork",
            "salmon",
            "tuna",
            "egg",
            "milk",
            "cheese",
            "yogurt",
            "butter",
            "rice",
            "bread",
            "pasta",
            "potato",
            "quinoa",
            "oats",
            "beans",
            "nuts",
            "olive oil",
            "salt",
            "pepper",
            "basil",
            "oregano",
            "thyme",
            "parsley",
            "flour",
            "sugar",
            "honey",
            "vinegar",
            "ginger",
        ]

        return config


def load_ingredient_config(
    config_path: Optional[str | Path] = None,
) -> IngredientConfig:
    """
    Load ingredient configuration from file or return default.

    Args:
        config_path: Path to configuration file. If None, uses default config.

    Returns:
        IngredientConfig instance
    """
    if config_path:
        return IngredientConfig.load_from_file(config_path)
    return IngredientConfig.get_default_config()


__all__ = ["IngredientConfig", "APIConfig", "ModelConfig", "load_ingredient_config"]
