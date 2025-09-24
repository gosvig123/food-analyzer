"""Lightweight inference pipeline for food recognition and ingredient detection."""

from .classification import FoodClassifier
from .core import (
    AnalyzedFood,
    Detection,
    FoodInferencePipeline,
    ImageInput,
    analyze_image,
)
from .detection import DepthEstimator, FoodDetector
from .utils import load_config, resolve_path_relative_to_project

__all__ = [
    "AnalyzedFood",
    "Detection",
    "ImageInput",
    "DepthEstimator",
    "FoodClassifier",
    "FoodDetector",
    "FoodInferencePipeline",
    "analyze_image",
    "load_config",
    "resolve_path_relative_to_project",
]
