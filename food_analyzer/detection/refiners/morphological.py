"""Morphological operations based mask refinement."""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
from PIL import Image

from .base import BaseMaskRefiner


class MorphologicalRefiner(BaseMaskRefiner):
    """Mask refiner using morphological operations.
    
    Lightweight alternative to SAM that works on CPU.
    Uses closing and opening operations to smooth mask edges.
    """
    
    def __init__(
        self,
        kernel_size: int = 5,
        iterations: int = 2,
    ):
        """Initialize morphological refiner.
        
        Args:
            kernel_size: Size of morphological kernel
            iterations: Number of morph operations to apply
        """
        self.kernel_size = max(1, kernel_size)
        self.iterations = max(0, iterations)
        self._cv2_available = self._check_cv2()
    
    def _check_cv2(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            warnings.warn("OpenCV not installed. Morphological refinement unavailable.")
            return False
    
    @property
    def available(self) -> bool:
        """Whether morphological refinement is available."""
        return self._cv2_available
    
    def refine(
        self,
        image: Image.Image,
        polygon: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Refine polygon using morphological operations."""
        if not self._cv2_available or not polygon:
            return polygon
        
        try:
            import cv2
            
            w, h = image.size
            
            # Convert polygon to mask
            pts = np.array(
                [[int(x), int(y)] for x, y in polygon], dtype=np.int32
            ).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
            )
            
            if self.iterations > 0:
                # Close to fill small holes
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, kernel, iterations=self.iterations
                )
                # Open to remove small protrusions
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, kernel, iterations=self.iterations
                )
            
            return self._mask_to_polygon(mask)
            
        except Exception as exc:
            warnings.warn(f"Morphological refinement failed: {exc}")
            return polygon
