"""Detection and segmentation components for food items."""

from .depth import DepthEstimator
from .detector import FoodDetector

__all__ = ["FoodDetector", "DepthEstimator"]
