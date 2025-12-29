"""SAM (Segment Anything Model) based mask refinement."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .base import BaseMaskRefiner


class SAMRefiner(BaseMaskRefiner):
    """Mask refiner using Segment Anything Model.
    
    Provides high-quality mask refinement but requires GPU and SAM checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "vit_b",
        device: str | None = None,
    ):
        """Initialize SAM refiner.
        
        Args:
            checkpoint_path: Path to SAM checkpoint file
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            device: Device to run on
        """
        import torch
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._predictor = None
        self._available = False
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize SAM predictor."""
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            warnings.warn("segment_anything not installed. SAM refinement unavailable.")
            return
        
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            warnings.warn(f"SAM checkpoint not found: {self.checkpoint_path}")
            return
        
        try:
            sam = sam_model_registry[self.model_type](
                checkpoint=str(self.checkpoint_path)
            )
            sam.to(self.device)
            self._predictor = SamPredictor(sam)
            self._available = True
        except Exception as exc:
            warnings.warn(f"Failed to initialize SAM: {exc}")
    
    @property
    def available(self) -> bool:
        """Whether SAM refinement is available."""
        return self._available
    
    def refine(
        self,
        image: Image.Image,
        polygon: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Refine polygon using SAM.
        
        Uses the polygon's bounding box as a prompt for SAM.
        """
        if not self._available or self._predictor is None:
            return polygon
        
        if not polygon:
            return polygon
        
        try:
            # Compute bounding box from polygon
            xs = [float(x) for x, _ in polygon]
            ys = [float(y) for _, y in polygon]
            x1, y1, x2, y2 = max(0.0, min(xs)), max(0.0, min(ys)), max(xs), max(ys)
            
            # Prepare image for SAM
            im = np.array(image.convert("RGB"))
            self._predictor.set_image(im)
            
            # Use multimask for better accuracy
            masks, scores, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([x1, y1, x2, y2])[None, :],
                multimask_output=True,
            )
            
            # Select best mask by score
            best_idx = np.argmax(scores)
            mask = masks[best_idx].astype(np.uint8) * 255
            
            return self._mask_to_polygon(mask)
            
        except Exception as exc:
            warnings.warn(f"SAM refinement failed: {exc}")
            return polygon
