"""Refactored food detector using modular backends and refiners.

This is the new architecture that separates concerns:
- Backends: Handle model inference (MaskRCNN, Hybrid, etc.)
- Refiners: Handle mask post-processing (SAM, Morphological)
- FoodDetectorV2: Coordinates backends and refiners
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

from PIL import Image

from ..core.types import Detection
from .backends import DetectorBackend, HybridBackend, MaskRCNNBackend
from .refiners import MaskRefiner, MorphologicalRefiner, SAMRefiner


class FoodDetectorV2:
    """Modular food detector with pluggable backends and refiners.
    
    This is a cleaner architecture that separates:
    - Detection backend (model inference)
    - Mask refinement (post-processing)
    
    Example:
        # Using default configuration
        detector = FoodDetectorV2()
        detections = detector(image)
        
        # Custom backend and refiner
        backend = HybridBackend(score_threshold=0.1)
        refiner = SAMRefiner(checkpoint_path="weights/sam.pth")
        detector = FoodDetectorV2(backend=backend, refiner=refiner)
    """
    
    def __init__(
        self,
        backend: DetectorBackend | None = None,
        refiner: MaskRefiner | None = None,
        # Convenience params for default backend creation
        backend_type: str = "hybrid",
        score_threshold: float = 0.25,
        max_detections: int = 100,
        device: str | None = None,
        semantic_food_classes: dict[str, int] | None = None,
        # Convenience params for default refiner creation
        refine_masks: bool = True,
        refine_method: str = "morph",
        sam_checkpoint: str | None = None,
        sam_model_type: str = "vit_b",
        morph_kernel: int = 5,
        morph_iters: int = 2,
    ):
        """Initialize the detector.
        
        Args:
            backend: Pre-configured detection backend (optional)
            refiner: Pre-configured mask refiner (optional)
            backend_type: Type of backend to create if not provided ('hybrid', 'mask_rcnn')
            score_threshold: Detection confidence threshold
            max_detections: Maximum detections to return
            device: Device to run on ('cuda' or 'cpu')
            semantic_food_classes: COCO class IDs for semantic segmentation
            refine_masks: Whether to refine masks
            refine_method: Refinement method ('sam' or 'morph')
            sam_checkpoint: Path to SAM checkpoint (for SAM refinement)
            sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            morph_kernel: Morphological kernel size
            morph_iters: Morphological iterations
        """
        self.refine_masks = refine_masks
        self.max_detections = max_detections
        
        # Create or use provided backend
        if backend is not None:
            self.backend = backend
        else:
            food_classes = None
            if semantic_food_classes:
                food_classes = set(semantic_food_classes.values())
            
            if backend_type == "hybrid":
                self.backend = HybridBackend(
                    device=device,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    semantic_food_classes=food_classes,
                )
            else:
                self.backend = MaskRCNNBackend(
                    device=device,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                )
        
        # Create or use provided refiner
        if refiner is not None:
            self.refiner = refiner
        elif not refine_masks:
            self.refiner = None
        elif refine_method == "sam" and sam_checkpoint:
            self.refiner = SAMRefiner(
                checkpoint_path=sam_checkpoint,
                model_type=sam_model_type,
                device=device,
            )
            # Fall back to morphological if SAM not available
            if not self.refiner.available:
                self.refiner = MorphologicalRefiner(
                    kernel_size=morph_kernel,
                    iterations=morph_iters,
                )
        else:
            self.refiner = MorphologicalRefiner(
                kernel_size=morph_kernel,
                iterations=morph_iters,
            )
    
    def __call__(self, image: Image.Image) -> List[Detection]:
        """Run detection on an image.
        
        Args:
            image: PIL Image to detect food in
            
        Returns:
            List of Detection objects
        """
        image = image.convert("RGB")
        
        # Run backend detection
        raw_results = self.backend.detect(image)
        
        # Convert to Detection objects and optionally refine
        detections = []
        for result in raw_results:
            mask_polygon = result.mask_polygon
            
            # Refine mask if enabled
            if self.refiner is not None and mask_polygon:
                try:
                    mask_polygon = self.refiner.refine(image, mask_polygon)
                except Exception:
                    pass  # Keep original polygon on failure
            
            detections.append(Detection(
                box=result.box,
                confidence=result.confidence,
                label=result.label,
                mask_polygon=mask_polygon,
            ))
        
        # Sort by confidence and limit
        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections[:self.max_detections]
    
    @classmethod
    def from_config(cls, config: dict) -> "FoodDetectorV2":
        """Create detector from configuration dict.
        
        Args:
            config: Configuration dictionary (detector section of config.json)
            
        Returns:
            Configured FoodDetectorV2 instance
        """
        backend_type = config.get("backend", "mask_rcnn_deeplabv3")
        if backend_type == "mask_rcnn_deeplabv3":
            backend_type = "hybrid"
        
        return cls(
            backend_type=backend_type,
            score_threshold=float(config.get("score_threshold", 0.25)),
            max_detections=int(config.get("max_detections", 100)),
            device=config.get("device"),
            semantic_food_classes=config.get("semantic_food_classes"),
            refine_masks=bool(config.get("refine_masks", True)),
            refine_method=str(config.get("refine_method", "morph")),
            sam_checkpoint=config.get("sam_checkpoint"),
            sam_model_type=config.get("sam_model_type", "vit_b"),
            morph_kernel=int(config.get("morph_kernel", 5)),
            morph_iters=int(config.get("morph_iters", 2)),
        )


# Alias for backward compatibility
ModularFoodDetector = FoodDetectorV2
