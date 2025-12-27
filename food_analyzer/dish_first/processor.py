"""Consolidated dish-first processing pipeline.

Combines dish classification, prior-based refinement, and detection reweighting
into a single processor class for cleaner integration with the main pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from PIL import Image


@dataclass
class DishFirstResult:
    """Result of dish-first processing."""
    detections: List[Dict[str, Any]]
    dish_topk: List[Dict[str, float]] = field(default_factory=list)
    prior_map: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DishFirstProcessor:
    """Consolidated processor for dish-first detection reweighting.
    
    Orchestrates:
    1. Dish classification (CLIP-based)
    2. Dish posterior refinement using observed ingredients
    3. Detection reweighting based on dish-ingredient priors
    4. Optional grams allocation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ingredient_labels: List[str],
    ) -> None:
        """Initialize the processor from config.
        
        Args:
            config: The dish_first section of the config
            ingredient_labels: List of ingredient labels from classifier
        """
        self.config = config
        self.ingredient_labels = ingredient_labels
        self.enabled = bool(config.get("enabled", False))
        
        self._dish_classifier = None
        self._priors_helper = None
        self._initialized = False
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self) -> None:
        """Lazy initialization of dish classifier and priors."""
        if self._initialized:
            return
            
        try:
            from .dish_classifier import DishClassifier
            from .priors import DishIngredientPriors
            
            # Initialize dish classifier
            dish_clf_cfg = self.config.get("dish_classifier", {})
            self._dish_classifier = DishClassifier.from_config(dish_clf_cfg)
            
            # Initialize priors helper
            priors_cfg = self.config.get("priors", {})
            self._priors_helper = DishIngredientPriors.from_config(
                priors_cfg, 
                ingredient_labels=self.ingredient_labels
            )
            
            self._initialized = True
            
        except Exception as exc:
            print(f"Dish-first initialization failed: {exc}")
            self.enabled = False
    
    def process(
        self,
        image: Image.Image | Path | str,
        detections: List[Dict[str, Any]],
        volume_config: Dict[str, Any] | None = None,
    ) -> DishFirstResult:
        """Process an image with dish-first pipeline.
        
        Args:
            image: PIL Image or path to image
            detections: Initial detection results from the pipeline
            volume_config: Optional volume/grams config for portion estimation
            
        Returns:
            DishFirstResult with reweighted detections and metadata
        """
        if not self.enabled or self._dish_classifier is None:
            return DishFirstResult(detections=detections)
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        try:
            # Step 1: Classify dish
            dish_topk = self._dish_classifier(image)
            
            # Step 2: Optionally refine dish posterior using observed ingredients
            dish_topk = self._refine_dish_posterior(dish_topk, detections)
            
            # Step 3: Build mixture prior over ingredients
            prior_map = self._build_prior_map(dish_topk)
            
            # Step 4: Reweight detections
            reweighted = self._reweight_detections(detections, prior_map)
            
            # Step 5: Optional grams allocation
            if volume_config:
                reweighted = self._allocate_grams(reweighted, volume_config)
            
            # Build metadata
            reweight_cfg = self.config.get("reweighting", {})
            metadata = {
                "dish_topk": dish_topk,
                "reweighting": {
                    "method": reweight_cfg.get("method", "boost"),
                    "alpha": reweight_cfg.get("alpha", 0.8),
                    "temperature": reweight_cfg.get("temperature", 1.0),
                },
            }
            
            return DishFirstResult(
                detections=reweighted,
                dish_topk=dish_topk,
                prior_map=prior_map,
                metadata=metadata,
            )
            
        except Exception as exc:
            print(f"Dish-first processing failed: {exc}")
            return DishFirstResult(detections=detections)
    
    def _refine_dish_posterior(
        self,
        dish_topk: List[Dict[str, float]],
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """Refine dish posterior using observed ingredient detections."""
        refine_cfg = self.config.get("refine", {})
        if not refine_cfg.get("enabled", True) or not dish_topk:
            return dish_topk
        
        try:
            from .refine import refine_dish_posterior
            
            priors_table = (
                self._priors_helper.table 
                if self._priors_helper and hasattr(self._priors_helper, "table") 
                else None
            )
            
            return refine_dish_posterior(
                dish_topk=dish_topk,
                detections=detections,
                priors=priors_table,
                ingredient_space=self.ingredient_labels,
                alpha_like=float(refine_cfg.get("alpha_like", 1.0)),
                max_topk=int(refine_cfg.get("max_topk", 5)),
            )
        except Exception:
            return dish_topk
    
    def _build_prior_map(
        self,
        dish_topk: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Build mixture prior over ingredient labels from dish posteriors."""
        if not dish_topk:
            # Uniform fallback
            n = max(1, len(self.ingredient_labels))
            return {lbl.lower(): 1.0 / n for lbl in self.ingredient_labels}
        
        try:
            from .priors import build_mixture_prior
            return build_mixture_prior(
                dish_topk, 
                self._priors_helper, 
                self.ingredient_labels
            )
        except Exception:
            n = max(1, len(self.ingredient_labels))
            return {lbl.lower(): 1.0 / n for lbl in self.ingredient_labels}
    
    def _reweight_detections(
        self,
        detections: List[Dict[str, Any]],
        prior_map: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Reweight and rerank detections based on prior map."""
        try:
            from .reweight import reweight_and_rerank_detections
            
            reweight_cfg = self.config.get("reweighting", {})
            return reweight_and_rerank_detections(detections, prior_map, reweight_cfg)
        except Exception:
            return detections
    
    def _allocate_grams(
        self,
        detections: List[Dict[str, Any]],
        volume_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Allocate gram weights to detections."""
        grams_cfg = self.config.get("grams", {})
        if not grams_cfg.get("enabled", False):
            return detections
        
        try:
            from .grams import allocate_grams_per_ingredient
            
            # Determine total grams
            total_source = grams_cfg.get("total_source", "fixed")
            if total_source == "fixed":
                total_grams = float(grams_cfg.get("fixed_total_grams", 
                    volume_config.get("grams_for_full_plate", 350.0)))
            elif total_source == "metadata":
                total_grams = float(volume_config.get("grams_for_full_plate", 350.0))
            elif total_source == "none":
                total_grams = None
            else:
                total_grams = float(volume_config.get("grams_for_full_plate", 350.0))
            
            grams_map = allocate_grams_per_ingredient(
                detections=detections,
                total_grams=total_grams,
                blend_lambda=float(grams_cfg.get("blend_lambda", 0.3)),
                beta=float(grams_cfg.get("area_exponent_beta", 0.5)),
                gamma=float(grams_cfg.get("confidence_exponent_gamma", 1.0)),
                min_grams=float(grams_cfg.get("min_grams", 0.0)),
                rounding=str(grams_cfg.get("rounding", "nearest_gram")),
            )
            
            # Inject grams into detections
            for det in detections:
                lbl = str(det.get("label", "")).strip().lower()
                if lbl in grams_map:
                    det["grams"] = float(grams_map[lbl])
            
            return detections
            
        except Exception:
            return detections
    
    @property
    def top_dish(self) -> str | None:
        """Get the top predicted dish label (for logging)."""
        return None  # Set after processing
    
    def get_top_dish_info(self, result: DishFirstResult) -> str:
        """Get a formatted string about the top dish prediction."""
        if not result.dish_topk:
            return ""
        top = result.dish_topk[0]
        return f"Dish-first: top dish {top.get('label')} @ {float(top.get('confidence', 0.0)):.2f}"


# Factory function for easy creation
def create_dish_first_processor(
    config: Dict[str, Any],
    ingredient_labels: List[str],
) -> DishFirstProcessor | None:
    """Create a DishFirstProcessor if enabled in config.
    
    Args:
        config: Full config dict (will extract dish_first section)
        ingredient_labels: List of ingredient labels from classifier
        
    Returns:
        DishFirstProcessor if enabled, None otherwise
    """
    dish_cfg = config.get("dish_first", {})
    if not isinstance(dish_cfg, dict):
        return None
    
    if not dish_cfg.get("enabled", False):
        return None
    
    return DishFirstProcessor(dish_cfg, ingredient_labels)
