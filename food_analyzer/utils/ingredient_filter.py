"""Post-processing filters to ensure only actual ingredients are reported."""

from __future__ import annotations

from typing import Dict, List

from ..utils.config import IngredientConfig, load_ingredient_config


class IngredientFilter:
    """Filter out non-ingredient labels from detection results."""

    def __init__(self, config: IngredientConfig | None = None):
        self.config = config or load_ingredient_config()

        # Build comprehensive filter lists
        self.non_ingredient_terms = set(
            getattr(self.config, "non_ingredient_terms", []) + self.config.generic_terms
        )

        # Enhanced filter patterns with better precision
        self.filter_patterns = {
            # Containers and utensils - NEVER ingredients
            "bowl", "soup bowl", "plate", "dish", "cup", "glass", "bottle",
            "container", "spoon", "fork", "knife", "utensil", "pot", "pan",
            "tray", "platter", "mug", "pitcher", "jar", "cutting board",
            "chopping board", "milk can", "eggnog",
            
            # Prepared dishes (not individual ingredients)
            "pizza", "burger", "sandwich", "cheeseburger", "hot dog",
            "burrito", "quesadilla",
            
            # Animals and non-food living things - HARD BLOCK
            "goldfish", "anemone fish", "lionfish", "fish",
            "butterfly", "sulphur butterfly", "cabbage butterfly",
            "prairie chicken", "crayfish", "starfish", "jellyfish",
            "leatherback turtle", "sea anemone", "coral", "sponge",
            
            # Dinosaurs and extinct creatures - HARD BLOCK
            "triceratops", "tyrannosaurus", "stegosaurus", "velociraptor",
            "brachiosaurus", "pterodactyl",
            
            # Objects and non-food items - HARD BLOCK
            "oil filter", "coil", "toilet seat", "toilet tissue", "toilet paper",
            "seat", "tissue", "paper towel", "napkin",
            
            # Machinery/Industrial
            "can", "milk can", "oil can", "filter",
            
            # Generic/vague terms
            "food", "item", "product", "meal", "snack", "beverage",
            "drink", "liquid", "solid", "mixture", "unknown",
            
            # Background/context items
            "table", "surface", "background", "texture", "pattern",
        }

    def is_valid_ingredient(self, label: str) -> bool:
        """Check if a label represents a valid ingredient."""
        if not label:
            return False

        label_lower = label.lower().strip()

        # Filter out non-ingredient terms
        for term in self.filter_patterns:
            if term in label_lower or label_lower in term:
                return False

        # Enhanced sauce handling - keep specific and identifiable sauces
        if "sauce" in label_lower:
            valid_sauces = {
                "soy sauce",
                "hot sauce",
                "tomato sauce",
                "chocolate sauce",
                "teriyaki sauce",
                "barbecue sauce",
                "bbq sauce",
                "pesto sauce",
                "marinara sauce",
                "alfredo sauce",
                "hollandaise sauce",
                "sriracha sauce",
                "worcestershire sauce",
                "tahini sauce",
            }
            valid_sauce_keywords = [
                "soy",
                "tomato",
                "chocolate",
                "chili",
                "sriracha",
                "teriyaki",
                "barbecue",
                "bbq",
                "pesto",
                "marinara",
                "alfredo",
                "hollandaise",
                "worcestershire",
                "tahini",
                "garlic",
                "curry",
                "cheese",
                "cream",
            ]

            if label_lower not in valid_sauces and not any(
                keyword in label_lower for keyword in valid_sauce_keywords
            ):
                return False

        # Enhanced generic term filtering
        generic_single_words = {
            "sauce",
            "seasoning",
            "spice",
            "herb",
            "oil",
            "vinegar",
            "powder",
            "extract",
            "paste",
            "liquid",
            "solid",
        }
        if label_lower in generic_single_words and len(label_lower.split()) == 1:
            return False

        # Filter out cooking methods without ingredient names
        cooking_methods = {
            "grilled",
            "fried",
            "baked",
            "roasted",
            "steamed",
            "boiled",
            "sautéed",
            "marinated",
            "seasoned",
            "chopped",
            "diced",
            "sliced",
        }
        if label_lower in cooking_methods:
            return False

        return True

    def filter_results(self, results: List[Dict]) -> List[Dict]:
        """Filter detection results to keep only valid ingredients."""
        filtered = []

        for result in results:
            label = result.get("label", "")
            if self.is_valid_ingredient(label):
                filtered.append(result)

        return filtered

    def filter_aggregated_results(self, aggregated: List[Dict]) -> List[Dict]:
        """Filter aggregated results to keep only valid ingredients."""
        filtered = []

        for result in aggregated:
            label = result.get("label", "")
            if self.is_valid_ingredient(label):
                filtered.append(result)

        return filtered


def filter_ingredients(
    results: List[Dict], config: IngredientConfig | None = None
) -> List[Dict]:
    """Convenience function to filter ingredient results."""
    filter_obj = IngredientFilter(config)
    return filter_obj.filter_results(results)


__all__ = ["IngredientFilter", "filter_ingredients"]
