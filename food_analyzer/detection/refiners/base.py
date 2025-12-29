"""Base protocol for mask refiners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class MaskRefiner(Protocol):
    """Protocol for mask refinement.
    
    Refiners take a rough polygon/mask and produce a more accurate version.
    """
    
    def refine(
        self,
        image: Image.Image,
        polygon: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Refine a polygon mask.
        
        Args:
            image: Original image
            polygon: List of (x, y) points defining the mask
            
        Returns:
            Refined polygon points
        """
        ...


class BaseMaskRefiner(ABC):
    """Abstract base class for mask refiners."""
    
    @abstractmethod
    def refine(
        self,
        image: Image.Image,
        polygon: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Refine a polygon mask."""
        pass
    
    def _polygon_to_mask(
        self,
        polygon: List[Tuple[float, float]],
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Convert polygon to binary mask."""
        try:
            import cv2
            w, h = size
            pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            return mask
        except ImportError:
            return np.zeros((size[1], size[0]), dtype=np.uint8)
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        """Convert binary mask to polygon."""
        try:
            import cv2
        except ImportError:
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
