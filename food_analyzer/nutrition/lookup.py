"""Nutrition lookup using USDA FoodData Central API.

Provides calorie, protein, carb, and fat information per 100g for ingredients.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class NutritionInfo:
    """Nutritional information for an ingredient (per 100g)."""
    
    ingredient: str
    calories: float = 0.0  # kcal per 100g
    protein: float = 0.0   # grams per 100g
    carbs: float = 0.0     # grams per 100g
    fat: float = 0.0       # grams per 100g
    fiber: float = 0.0     # grams per 100g
    sugar: float = 0.0     # grams per 100g
    sodium: float = 0.0    # mg per 100g
    source: str = "unknown"
    fdc_id: int | None = None
    
    def calories_for_grams(self, grams: float) -> float:
        """Calculate calories for a given portion in grams."""
        return (self.calories * grams) / 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ingredient": self.ingredient,
            "calories_per_100g": self.calories,
            "protein_per_100g": self.protein,
            "carbs_per_100g": self.carbs,
            "fat_per_100g": self.fat,
            "fiber_per_100g": self.fiber,
            "sugar_per_100g": self.sugar,
            "sodium_per_100g": self.sodium,
            "source": self.source,
            "fdc_id": self.fdc_id,
        }


# Common ingredient nutrition (fallback when API unavailable)
# Values are per 100g, sourced from USDA averages
_DEFAULT_NUTRITION: Dict[str, Dict[str, float]] = {
    # Vegetables
    "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2, "fiber": 1.2},
    "lettuce": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2, "fiber": 1.3},
    "cucumber": {"calories": 16, "protein": 0.7, "carbs": 3.6, "fat": 0.1, "fiber": 0.5},
    "carrot": {"calories": 41, "protein": 0.9, "carbs": 9.6, "fat": 0.2, "fiber": 2.8},
    "onion": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7},
    "pepper": {"calories": 31, "protein": 1.0, "carbs": 6.0, "fat": 0.3, "fiber": 2.1},
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7.0, "fat": 0.4, "fiber": 2.6},
    "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2},
    "potato": {"calories": 77, "protein": 2.0, "carbs": 17.5, "fat": 0.1, "fiber": 2.2},
    "mushroom": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3, "fiber": 1.0},
    
    # Fruits
    "apple": {"calories": 52, "protein": 0.3, "carbs": 13.8, "fat": 0.2, "fiber": 2.4},
    "banana": {"calories": 89, "protein": 1.1, "carbs": 22.8, "fat": 0.3, "fiber": 2.6},
    "orange": {"calories": 47, "protein": 0.9, "carbs": 11.8, "fat": 0.1, "fiber": 2.4},
    "strawberry": {"calories": 32, "protein": 0.7, "carbs": 7.7, "fat": 0.3, "fiber": 2.0},
    "grape": {"calories": 69, "protein": 0.7, "carbs": 18.1, "fat": 0.2, "fiber": 0.9},
    
    # Proteins
    "chicken": {"calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0},
    "beef": {"calories": 250, "protein": 26.0, "carbs": 0.0, "fat": 15.0, "fiber": 0.0},
    "pork": {"calories": 242, "protein": 27.0, "carbs": 0.0, "fat": 14.0, "fiber": 0.0},
    "fish": {"calories": 206, "protein": 22.0, "carbs": 0.0, "fat": 12.0, "fiber": 0.0},
    "salmon": {"calories": 208, "protein": 20.0, "carbs": 0.0, "fat": 13.0, "fiber": 0.0},
    "egg": {"calories": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "fiber": 0.0},
    "tofu": {"calories": 76, "protein": 8.0, "carbs": 1.9, "fat": 4.8, "fiber": 0.3},
    
    # Grains & Starches
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28.2, "fat": 0.3, "fiber": 0.4},
    "pasta": {"calories": 131, "protein": 5.0, "carbs": 25.0, "fat": 1.1, "fiber": 1.8},
    "bread": {"calories": 265, "protein": 9.0, "carbs": 49.0, "fat": 3.2, "fiber": 2.7},
    "noodle": {"calories": 138, "protein": 4.5, "carbs": 25.0, "fat": 2.1, "fiber": 1.2},
    
    # Dairy
    "cheese": {"calories": 402, "protein": 25.0, "carbs": 1.3, "fat": 33.0, "fiber": 0.0},
    "milk": {"calories": 42, "protein": 3.4, "carbs": 5.0, "fat": 1.0, "fiber": 0.0},
    "yogurt": {"calories": 59, "protein": 10.0, "carbs": 3.6, "fat": 0.7, "fiber": 0.0},
    "butter": {"calories": 717, "protein": 0.9, "carbs": 0.1, "fat": 81.0, "fiber": 0.0},
    
    # Sauces & Condiments
    "sauce": {"calories": 50, "protein": 1.0, "carbs": 10.0, "fat": 0.5, "fiber": 0.5},
    "ketchup": {"calories": 112, "protein": 1.7, "carbs": 26.0, "fat": 0.1, "fiber": 0.3},
    "mayonnaise": {"calories": 680, "protein": 1.0, "carbs": 0.6, "fat": 75.0, "fiber": 0.0},
    "olive oil": {"calories": 884, "protein": 0.0, "carbs": 0.0, "fat": 100.0, "fiber": 0.0},
    
    # Common dishes (approximate)
    "pizza": {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0, "fiber": 2.3},
    "burger": {"calories": 295, "protein": 17.0, "carbs": 24.0, "fat": 14.0, "fiber": 1.3},
    "salad": {"calories": 20, "protein": 1.5, "carbs": 3.5, "fat": 0.2, "fiber": 1.8},
    "soup": {"calories": 35, "protein": 2.0, "carbs": 5.0, "fat": 1.0, "fiber": 0.8},
}


class NutritionLookup:
    """Lookup nutritional information for ingredients via USDA API with caching."""
    
    # USDA nutrient IDs for key nutrients
    NUTRIENT_IDS = {
        "energy": 1008,      # Energy (kcal)
        "protein": 1003,     # Protein
        "carbs": 1005,       # Carbohydrates
        "fat": 1004,         # Total fat
        "fiber": 1079,       # Fiber
        "sugar": 2000,       # Total sugars
        "sodium": 1093,      # Sodium
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        use_cache: bool = True,
    ):
        """Initialize nutrition lookup.
        
        Args:
            api_key: USDA FoodData Central API key (optional, uses DEMO_KEY if not provided)
            cache_path: Path to cache file
            use_cache: Whether to use local cache
        """
        self.api_key = api_key or "DEMO_KEY"
        self.cache_path = Path(cache_path) if cache_path else Path(".nutrition_cache.json")
        self.use_cache = use_cache
        self._cache: Dict[str, NutritionInfo] = {}
        
        if use_cache:
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load nutrition cache from disk."""
        if not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            for key, info in data.items():
                self._cache[key] = NutritionInfo(
                    ingredient=info.get("ingredient", key),
                    calories=info.get("calories_per_100g", 0),
                    protein=info.get("protein_per_100g", 0),
                    carbs=info.get("carbs_per_100g", 0),
                    fat=info.get("fat_per_100g", 0),
                    fiber=info.get("fiber_per_100g", 0),
                    sugar=info.get("sugar_per_100g", 0),
                    sodium=info.get("sodium_per_100g", 0),
                    source=info.get("source", "cache"),
                    fdc_id=info.get("fdc_id"),
                )
        except Exception as exc:
            warnings.warn(f"Failed to load nutrition cache: {exc}")
    
    def _save_cache(self) -> None:
        """Save nutrition cache to disk."""
        if not self.use_cache:
            return
        try:
            data = {key: info.to_dict() for key, info in self._cache.items()}
            self.cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            warnings.warn(f"Failed to save nutrition cache: {exc}")
    
    def _normalize_ingredient(self, ingredient: str) -> str:
        """Normalize ingredient name for lookup."""
        return ingredient.lower().strip()
    
    def get(self, ingredient: str) -> NutritionInfo:
        """Get nutrition info for an ingredient.
        
        Args:
            ingredient: Ingredient name
            
        Returns:
            NutritionInfo with nutritional data
        """
        key = self._normalize_ingredient(ingredient)
        
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Try local defaults
        if key in _DEFAULT_NUTRITION:
            info = NutritionInfo(
                ingredient=ingredient,
                source="default",
                **_DEFAULT_NUTRITION[key]
            )
            self._cache[key] = info
            return info
        
        # Try partial match in defaults
        for default_key, values in _DEFAULT_NUTRITION.items():
            if default_key in key or key in default_key:
                info = NutritionInfo(
                    ingredient=ingredient,
                    source="default_partial",
                    **values
                )
                self._cache[key] = info
                return info
        
        # Try USDA API
        try:
            info = self._fetch_from_usda(ingredient)
            if info:
                self._cache[key] = info
                self._save_cache()
                return info
        except Exception as exc:
            warnings.warn(f"USDA API lookup failed for {ingredient}: {exc}")
        
        # Return empty info as fallback
        return NutritionInfo(ingredient=ingredient, source="unknown")
    
    def _fetch_from_usda(self, ingredient: str) -> NutritionInfo | None:
        """Fetch nutrition data from USDA FoodData Central API."""
        try:
            # Search for the food
            search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                "api_key": self.api_key,
                "query": ingredient,
                "pageSize": 5,
                "dataType": ["Foundation", "SR Legacy"],
            }
            
            query_string = urllib.parse.urlencode(params, doseq=True)
            url = f"{search_url}?{query_string}"
            
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json")
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            foods = data.get("foods", [])
            if not foods:
                return None
            
            # Use first result
            food = foods[0]
            fdc_id = food.get("fdcId")
            
            # Extract nutrients
            nutrients = {}
            for nutrient in food.get("foodNutrients", []):
                nutrient_id = nutrient.get("nutrientId")
                value = nutrient.get("value", 0)
                
                if nutrient_id == self.NUTRIENT_IDS["energy"]:
                    nutrients["calories"] = value
                elif nutrient_id == self.NUTRIENT_IDS["protein"]:
                    nutrients["protein"] = value
                elif nutrient_id == self.NUTRIENT_IDS["carbs"]:
                    nutrients["carbs"] = value
                elif nutrient_id == self.NUTRIENT_IDS["fat"]:
                    nutrients["fat"] = value
                elif nutrient_id == self.NUTRIENT_IDS["fiber"]:
                    nutrients["fiber"] = value
                elif nutrient_id == self.NUTRIENT_IDS["sugar"]:
                    nutrients["sugar"] = value
                elif nutrient_id == self.NUTRIENT_IDS["sodium"]:
                    nutrients["sodium"] = value
            
            return NutritionInfo(
                ingredient=ingredient,
                calories=nutrients.get("calories", 0),
                protein=nutrients.get("protein", 0),
                carbs=nutrients.get("carbs", 0),
                fat=nutrients.get("fat", 0),
                fiber=nutrients.get("fiber", 0),
                sugar=nutrients.get("sugar", 0),
                sodium=nutrients.get("sodium", 0),
                source="usda",
                fdc_id=fdc_id,
            )
            
        except Exception as exc:
            warnings.warn(f"USDA API error: {exc}")
            return None
    
    def get_batch(self, ingredients: List[str]) -> Dict[str, NutritionInfo]:
        """Get nutrition info for multiple ingredients.
        
        Args:
            ingredients: List of ingredient names
            
        Returns:
            Dict mapping ingredient names to NutritionInfo
        """
        return {ing: self.get(ing) for ing in ingredients}


# Singleton instance for convenience
_default_lookup: NutritionLookup | None = None


def get_nutrition_for_ingredient(
    ingredient: str,
    api_key: str | None = None,
) -> NutritionInfo:
    """Convenience function to get nutrition for a single ingredient.
    
    Args:
        ingredient: Ingredient name
        api_key: Optional USDA API key
        
    Returns:
        NutritionInfo for the ingredient
    """
    global _default_lookup
    if _default_lookup is None:
        _default_lookup = NutritionLookup(api_key=api_key)
    return _default_lookup.get(ingredient)
