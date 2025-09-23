"""Classification components for identifying food types."""

from .classifier import FoodClassifier
from .ingredient_api import IngredientLabelFetcher, get_dynamic_ingredient_labels
from .intelligent_labels import (
    IntelligentLabelExtractor,
    get_intelligent_ingredient_labels,
)

__all__ = [
    "FoodClassifier",
    "IngredientLabelFetcher",
    "get_dynamic_ingredient_labels",
    "IntelligentLabelExtractor",
    "get_intelligent_ingredient_labels",
]
