"""Hybrid detection backend combining Mask R-CNN + DeepLabV3+."""
from __future__ import annotations

import warnings
from typing import List, Set

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import detection as detection_models
from torchvision.models import segmentation as segmentation_models

from .base import BaseDetectorBackend, DetectionResult


class HybridBackend(BaseDetectorBackend):
    """Hybrid backend combining instance and semantic segmentation.
    
    Uses Mask R-CNN for instance segmentation and DeepLabV3+ for
    semantic segmentation, combining results for better food detection coverage.
    """
    
    # Default COCO food class IDs for semantic segmentation
    DEFAULT_FOOD_CLASSES = {47, 48, 49, 50, 51, 52, 53, 54}
    
    def __init__(
        self,
        device: str | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        semantic_food_classes: Set[int] | None = None,
        enable_semantic: bool = True,
    ):
        """Initialize hybrid backend.
        
        Args:
            device: Device to run on
            score_threshold: Minimum confidence threshold
            max_detections: Maximum detections to return
            semantic_food_classes: COCO class IDs to detect with semantic seg
            enable_semantic: Whether to enable semantic segmentation
        """
        super().__init__(device, score_threshold, max_detections)
        
        self.semantic_food_classes = semantic_food_classes or self.DEFAULT_FOOD_CLASSES
        self.enable_semantic = enable_semantic and self._device.startswith("cuda")
        
        self._preprocess = transforms.Compose([transforms.ToTensor()])
        self._semantic_preprocess = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._instance_model = None
        self._semantic_model = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load detection models."""
        try:
            # Load Mask R-CNN for instance segmentation
            mask_weights = detection_models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self._instance_model = detection_models.maskrcnn_resnet50_fpn_v2(
                weights=mask_weights
            )
            self._instance_model.to(self._device).eval()
            self._categories = mask_weights.meta.get("categories", self._categories)
            
            # Load DeepLabV3+ for semantic segmentation (GPU only)
            if self.enable_semantic:
                semantic_weights = segmentation_models.DeepLabV3_ResNet50_Weights.DEFAULT
                self._semantic_model = segmentation_models.deeplabv3_resnet50(
                    weights=semantic_weights
                )
                self._semantic_model.to(self._device).eval()
                
        except Exception as exc:
            warnings.warn(f"Failed to load models: {exc}")
    
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """Run hybrid detection on an image."""
        if self._instance_model is None:
            return []
        
        image = image.convert("RGB")
        
        # Get instance segmentation results
        instance_results = self._detect_instances(image)
        
        # Get semantic segmentation results
        semantic_results = []
        if self._semantic_model is not None and self.enable_semantic:
            semantic_results = self._detect_semantic(image)
        
        # Combine results
        all_results = instance_results + semantic_results
        
        # Sort by confidence and limit
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        return all_results[:self.max_detections]
    
    def _detect_instances(self, image: Image.Image) -> List[DetectionResult]:
        """Run instance segmentation with Mask R-CNN."""
        tensor = self._preprocess(image).to(self._device)
        
        with torch.inference_mode():
            outputs = self._instance_model([tensor])[0]
        
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
            
            mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
            mask_polygon = self._mask_to_polygon(mask_binary)
            
            results.append(DetectionResult(
                box=(left, top, right, bottom),
                confidence=float(score),
                label=category,
                mask=mask_binary,
                mask_polygon=mask_polygon,
            ))
        
        return results
    
    def _detect_semantic(self, image: Image.Image) -> List[DetectionResult]:
        """Run semantic segmentation with DeepLabV3+."""
        try:
            import cv2
        except ImportError:
            return []
        
        img_tensor = self._preprocess(image).unsqueeze(0)
        img_tensor = self._semantic_preprocess(img_tensor).to(self._device)
        
        with torch.inference_mode():
            output = self._semantic_model(img_tensor)["out"]
            predictions = torch.softmax(output, dim=1)
            pred_classes = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
        
        results = []
        for food_class in self.semantic_food_classes:
            mask = (pred_classes == food_class).astype(np.uint8) * 255
            if mask.sum() < 100:  # Skip very small segments
                continue
            
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            mask_polygon = self._mask_to_polygon(mask)
            
            # Calculate confidence based on mask quality
            mask_area = cv2.contourArea(largest_contour)
            total_area = mask.shape[0] * mask.shape[1]
            area_ratio = mask_area / total_area
            bbox_area = w * h
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            base_confidence = 0.5 + (area_ratio * 0.3)
            if aspect_ratio > 3.0:
                base_confidence *= 0.7
            if 500 < bbox_area < 50000:
                base_confidence *= 1.1
            elif bbox_area < 200:
                base_confidence *= 0.5
            
            confidence = min(0.8, max(0.3, base_confidence))
            
            results.append(DetectionResult(
                box=(x, y, x + w, y + h),
                confidence=confidence,
                label="food",
                mask=mask,
                mask_polygon=mask_polygon,
            ))
        
        return results
