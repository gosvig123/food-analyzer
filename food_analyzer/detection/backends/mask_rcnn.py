"""Mask R-CNN detection backend."""
from __future__ import annotations

import warnings
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import detection as detection_models

from .base import BaseDetectorBackend, DetectionResult


class MaskRCNNBackend(BaseDetectorBackend):
    """Detection backend using Mask R-CNN for instance segmentation."""
    
    def __init__(
        self,
        device: str | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        use_v2: bool = True,
    ):
        """Initialize Mask R-CNN backend.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            score_threshold: Minimum confidence threshold
            max_detections: Maximum number of detections to return
            use_v2: Whether to use the v2 model weights (better accuracy)
        """
        super().__init__(device, score_threshold, max_detections)
        self.use_v2 = use_v2
        self._preprocess = transforms.Compose([transforms.ToTensor()])
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Mask R-CNN model."""
        try:
            if self.use_v2:
                weights = detection_models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self._model = detection_models.maskrcnn_resnet50_fpn_v2(weights=weights)
            else:
                weights = detection_models.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                self._model = detection_models.maskrcnn_resnet50_fpn(weights=weights)
            
            self._model.to(self._device).eval()
            self._categories = weights.meta.get("categories", self._categories)
        except Exception as exc:
            warnings.warn(f"Failed to load Mask R-CNN: {exc}")
            self._model = None
    
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """Run Mask R-CNN detection on an image."""
        if self._model is None:
            return []
        
        image = image.convert("RGB")
        tensor = self._preprocess(image).to(self._device)
        
        with torch.inference_mode():
            outputs = self._model([tensor])[0]
        
        results = []
        for score, label_idx, box, mask in zip(
            outputs.get("scores", []).cpu().numpy(),
            outputs.get("labels", []).cpu().numpy(),
            outputs.get("boxes", []).cpu().numpy(),
            outputs.get("masks", []).cpu().numpy(),
        ):
            if score < self.score_threshold:
                continue
            
            left, top, right, bottom = (int(round(v)) for v in box)
            category = (
                self._categories[label_idx]
                if 0 <= label_idx < len(self._categories)
                else "food"
            )
            
            # Convert mask to binary
            mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
            mask_polygon = self._mask_to_polygon(mask_binary)
            
            results.append(DetectionResult(
                box=(left, top, right, bottom),
                confidence=float(score),
                label=category,
                mask=mask_binary,
                mask_polygon=mask_polygon,
            ))
        
        # Sort by confidence and limit
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:self.max_detections]
