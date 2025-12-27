"""Nutrition estimation for detected food items.

Combines portion estimation with nutrition lookup to calculate
total nutritional values for detected foods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .lookup import NutritionLookup, NutritionInfo


@dataclass
class FoodNutrition:
    """Nutrition information for a detected food item with portion."""
    
    label: str
    grams: float
    calories: float
    protein: float
    carbs: float
    fat: float
    nutrition_per_100g: NutritionInfo | None = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "grams": self.grams,
            "calories": self.calories,
            "protein": self.protein,
            "carbs": self.carbs,
            "fat": self.fat,
        }


class NutritionEstimator:
    """Estimates nutrition for detected food items."""
    
    def __init__(
        self,
        lookup: NutritionLookup | None = None,
        default_portion_grams: float = 100.0,
    ):
        """Initialize estimator.
        
        Args:
            lookup: NutritionLookup instance (creates default if None)
            default_portion_grams: Default portion size when grams not provided
        """
        self.lookup = lookup or NutritionLookup()
        self.default_portion_grams = default_portion_grams
    
    def estimate_for_detection(
        self,
        detection: Dict[str, Any],
    ) -> FoodNutrition:
        """Estimate nutrition for a single detection.
        
        Args:
            detection: Detection dict with 'label' and optionally 'grams'
            
        Returns:
            FoodNutrition with estimated values
        """
        label = str(detection.get("label", "unknown"))
        grams = float(detection.get("grams", self.default_portion_grams))
        
        # Look up nutrition per 100g
        info = self.lookup.get(label)
        
        # Calculate for portion
        factor = grams / 100.0
        
        return FoodNutrition(
            label=label,
            grams=grams,
            calories=info.calories * factor,
            protein=info.protein * factor,
            carbs=info.carbs * factor,
            fat=info.fat * factor,
            nutrition_per_100g=info,
        )
    
    def estimate_for_detections(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[FoodNutrition]:
        """Estimate nutrition for multiple detections.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            List of FoodNutrition objects
        """
        return [self.estimate_for_detection(d) for d in detections]
    
    def estimate_totals(
        self,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate total nutrition across all detections.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            Dict with total calories, protein, carbs, fat, grams
        """
        items = self.estimate_for_detections(detections)
        
        return {
            "total_grams": sum(item.grams for item in items),
            "total_calories": sum(item.calories for item in items),
            "total_protein": sum(item.protein for item in items),
            "total_carbs": sum(item.carbs for item in items),
            "total_fat": sum(item.fat for item in items),
        }
    
    def enrich_detections(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add nutrition info to detection dicts in-place.
        
        Args:
            detections: List of detection dicts to enrich
            
        Returns:
            Same list with nutrition fields added
        """
        for detection in detections:
            nutrition = self.estimate_for_detection(detection)
            detection["calories"] = nutrition.calories
            detection["protein"] = nutrition.protein
            detection["carbs"] = nutrition.carbs
            detection["fat"] = nutrition.fat
        
        return detections
    
    def enrich_aggregates(
        self,
        aggregates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add nutrition info to aggregated results.
        
        Args:
            aggregates: List of aggregated result dicts with 'label' and 'grams'
            
        Returns:
            Same list with nutrition fields added
        """
        for entry in aggregates:
            label = str(entry.get("label", ""))
            grams = float(entry.get("grams", 0))
            
            if grams > 0:
                info = self.lookup.get(label)
                factor = grams / 100.0
                entry["calories"] = info.calories * factor
                entry["protein"] = info.protein * factor
                entry["carbs"] = info.carbs * factor
                entry["fat"] = info.fat * factor
            else:
                entry["calories"] = 0.0
                entry["protein"] = 0.0
                entry["carbs"] = 0.0
                entry["fat"] = 0.0
        
        return aggregates
