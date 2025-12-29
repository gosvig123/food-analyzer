"""Detection and segmentation components for food items.

This module provides modular detection with pluggable backends and refiners.

Quick Start:
    # Default detector (Mask R-CNN + DeepLabV3 hybrid)
    from food_analyzer.detection import FoodDetector
    detector = FoodDetector()
    detections = detector(image)
    
    # Modular detector with custom backend
    from food_analyzer.detection import FoodDetectorV2, YOLOBackend
    backend = YOLOBackend(model_path="weights/food_yolo.pt")
    detector = FoodDetectorV2(backend=backend)
    
    # Using SAM for mask refinement
    from food_analyzer.detection import FoodDetectorV2, SAMRefiner
    refiner = SAMRefiner(checkpoint_path="weights/sam.pth")
    detector = FoodDetectorV2(refiner=refiner)

Available Backends:
    - HybridBackend: Mask R-CNN + DeepLabV3 (default, best recall)
    - MaskRCNNBackend: Mask R-CNN only (faster)
    - YOLOBackend: YOLO models (supports custom food models)
    - CustomTorchBackend: Any PyTorch detection model

Available Refiners:
    - SAMRefiner: Segment Anything Model (best quality, requires GPU)
    - MorphologicalRefiner: OpenCV morphological ops (fast, CPU-friendly)
"""

from .depth import DepthEstimator
from .detector import FoodDetector
from .detector_v2 import FoodDetectorV2, ModularFoodDetector

# Modular components
from .backends import (
    DetectorBackend,
    MaskRCNNBackend,
    HybridBackend,
    YOLOBackend,
    CustomTorchBackend,
    create_backend_from_config,
)
from .refiners import MaskRefiner, SAMRefiner, MorphologicalRefiner

__all__ = [
    # Main detectors
    "FoodDetector",
    "FoodDetectorV2",
    "ModularFoodDetector",
    "DepthEstimator",
    # Backends
    "DetectorBackend",
    "MaskRCNNBackend",
    "HybridBackend",
    "YOLOBackend",
    "CustomTorchBackend",
    "create_backend_from_config",
    # Refiners
    "MaskRefiner",
    "SAMRefiner",
    "MorphologicalRefiner",
]
