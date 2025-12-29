"""Base protocol for detection backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    """Raw detection result from a backend."""
    
    box: Tuple[int, int, int, int]  # (left, top, right, bottom)
    confidence: float
    label: str
    mask: np.ndarray | None = None  # Binary mask if available
    mask_polygon: List[Tuple[float, float]] = field(default_factory=list)


@runtime_checkable
class DetectorBackend(Protocol):
    """Protocol for detection backends.
    
    Backends handle the actual model inference and return raw detections.
    Mask refinement is handled separately by Refiner classes.
    """
    
    @property
    def device(self) -> str:
        """Device the model runs on."""
        ...
    
    @property
    def categories(self) -> List[str]:
        """List of category names the model can detect."""
        ...
    
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """Run detection on an image.
        
        Args:
            image: PIL Image to detect objects in
            
        Returns:
            List of DetectionResult objects
        """
        ...


class BaseDetectorBackend(ABC):
    """Abstract base class for detector backends with common functionality."""
    
    def __init__(
        self,
        device: str | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
    ):
        import torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self._categories: List[str] = ["food"]
        self._model = None
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def categories(self) -> List[str]:
        return self._categories
    
    @abstractmethod
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """Run detection on an image."""
        pass
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Convert a binary mask to a polygon."""
        try:
            import cv2
        except ImportError:
            # Fallback to bounding box
            ys, xs = np.where(mask > 0)
            if xs.size == 0 or ys.size == 0:
                return []
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            return [
                (float(x1), float(y1)),
                (float(x2), float(y1)),
                (float(x2), float(y2)),
                (float(x1), float(y2)),
            ]
        
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        epsilon = 0.001 * max(1.0, peri) if len(cnt) > 500 else 0.002 * max(1.0, peri)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
        
        max_points = min(300, max(50, int(peri / 10)))
        step = max(1, int(len(poly) / max_points))
        return poly[::step]
