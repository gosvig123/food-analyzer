"""Custom/food-specific model backend.

Supports loading:
- YOLO models (via ultralytics)
- Custom torchvision models
- ONNX models

This enables using food-specific models for better detection accuracy.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image

from .base import BaseDetectorBackend, DetectionResult


class YOLOBackend(BaseDetectorBackend):
    """Backend for YOLO models (food detection, etc.).
    
    Supports ultralytics YOLO models including:
    - Pre-trained YOLOv8/v5 models
    - Custom fine-tuned food detection models
    
    Example:
        # Using a custom food model
        backend = YOLOBackend(
            model_path="weights/food_yolov8.pt",
            score_threshold=0.3,
        )
        
        # Using pre-trained model
        backend = YOLOBackend(model_path="yolov8n.pt")
    """
    
    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        device: str | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        imgsz: int = 640,
        food_classes_only: bool = True,
    ):
        """Initialize YOLO backend.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run on
            score_threshold: Confidence threshold
            max_detections: Maximum detections
            imgsz: Input image size
            food_classes_only: Filter to food classes only (for COCO models)
        """
        super().__init__(device, score_threshold, max_detections)
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.food_classes_only = food_classes_only
        
        # COCO food class indices (for filtering pre-trained models)
        self._coco_food_classes = {
            46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
            50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake",
        }
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            warnings.warn(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
            return
        
        try:
            self._model = YOLO(str(self.model_path))
            # Get class names from model
            if hasattr(self._model, "names"):
                self._categories = list(self._model.names.values())
        except Exception as exc:
            warnings.warn(f"Failed to load YOLO model: {exc}")
    
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """Run YOLO detection."""
        if self._model is None:
            return []
        
        try:
            # Run inference
            results = self._model(
                image,
                conf=self.score_threshold,
                imgsz=self.imgsz,
                device=self._device,
                verbose=False,
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Filter to food classes if using COCO model
                    if self.food_classes_only and cls_id not in self._coco_food_classes:
                        if self._categories == list(range(80)):  # COCO model
                            continue
                    
                    # Get class name
                    if cls_id < len(self._categories):
                        label = self._categories[cls_id]
                    elif cls_id in self._coco_food_classes:
                        label = self._coco_food_classes[cls_id]
                    else:
                        label = "food"
                    
                    # Get mask if available
                    mask = None
                    mask_polygon = []
                    if hasattr(result, "masks") and result.masks is not None:
                        try:
                            mask_data = result.masks.data[i].cpu().numpy()
                            mask = (mask_data > 0.5).astype(np.uint8) * 255
                            mask_polygon = self._mask_to_polygon(mask)
                        except Exception:
                            pass
                    
                    detections.append(DetectionResult(
                        box=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=conf,
                        label=label,
                        mask=mask,
                        mask_polygon=mask_polygon,
                    ))
            
            # Sort and limit
            detections.sort(key=lambda x: x.confidence, reverse=True)
            return detections[:self.max_detections]
            
        except Exception as exc:
            warnings.warn(f"YOLO inference failed: {exc}")
            return []


class CustomTorchBackend(BaseDetectorBackend):
    """Backend for custom PyTorch models.
    
    Supports loading custom detection models saved as:
    - torch.save() checkpoints
    - TorchScript models
    
    The model should output a dict with keys:
    - boxes: [N, 4] tensor of xyxy coordinates
    - scores: [N] tensor of confidence scores
    - labels: [N] tensor of class indices
    - masks: [N, H, W] tensor of masks (optional)
    """
    
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        class_names: List[str] | None = None,
    ):
        """Initialize custom torch backend.
        
        Args:
            model_path: Path to model checkpoint or TorchScript
            device: Device to run on
            score_threshold: Confidence threshold
            max_detections: Maximum detections
            class_names: List of class names
        """
        super().__init__(device, score_threshold, max_detections)
        self.model_path = Path(model_path)
        self._categories = class_names or ["food"]
        self._load_model()
    
    def _load_model(self) -> None:
        """Load custom model."""
        import torch
        
        if not self.model_path.exists():
            warnings.warn(f"Model not found: {self.model_path}")
            return
        
        try:
            # Try loading as TorchScript first
            if self.model_path.suffix == ".pt":
                try:
                    self._model = torch.jit.load(str(self.model_path))
                except Exception:
                    # Fall back to regular checkpoint
                    self._model = torch.load(str(self.model_path), map_location=self._device)
            else:
                self._model = torch.load(str(self.model_path), map_location=self._device)
            
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
            if hasattr(self._model, "eval"):
                self._model.eval()
                
        except Exception as exc:
            warnings.warn(f"Failed to load custom model: {exc}")
    
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """Run custom model detection."""
        if self._model is None:
            return []
        
        import torch
        from torchvision import transforms
        
        try:
            # Preprocess
            transform = transforms.Compose([transforms.ToTensor()])
            tensor = transform(image).unsqueeze(0).to(self._device)
            
            # Inference
            with torch.inference_mode():
                outputs = self._model(tensor)
            
            # Handle different output formats
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            detections = []
            boxes = outputs.get("boxes", torch.tensor([])).cpu().numpy()
            scores = outputs.get("scores", torch.tensor([])).cpu().numpy()
            labels = outputs.get("labels", torch.tensor([])).cpu().numpy()
            masks = outputs.get("masks", None)
            
            for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
                if score < self.score_threshold:
                    continue
                
                label = (
                    self._categories[label_idx]
                    if label_idx < len(self._categories)
                    else "food"
                )
                
                mask = None
                mask_polygon = []
                if masks is not None:
                    try:
                        mask = (masks[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
                        if mask.ndim == 3:
                            mask = mask[0]
                        mask_polygon = self._mask_to_polygon(mask)
                    except Exception:
                        pass
                
                detections.append(DetectionResult(
                    box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    confidence=float(score),
                    label=label,
                    mask=mask,
                    mask_polygon=mask_polygon,
                ))
            
            detections.sort(key=lambda x: x.confidence, reverse=True)
            return detections[:self.max_detections]
            
        except Exception as exc:
            warnings.warn(f"Custom model inference failed: {exc}")
            return []


def create_backend_from_config(config: dict) -> BaseDetectorBackend:
    """Factory function to create backend from config.
    
    Args:
        config: Backend configuration dict
        
    Returns:
        Configured backend instance
    """
    backend_type = config.get("type", "hybrid")
    
    if backend_type == "yolo":
        return YOLOBackend(
            model_path=config.get("model_path", "yolov8n.pt"),
            device=config.get("device"),
            score_threshold=float(config.get("score_threshold", 0.25)),
            max_detections=int(config.get("max_detections", 100)),
            imgsz=int(config.get("imgsz", 640)),
            food_classes_only=bool(config.get("food_classes_only", True)),
        )
    
    elif backend_type == "custom":
        return CustomTorchBackend(
            model_path=config.get("model_path"),
            device=config.get("device"),
            score_threshold=float(config.get("score_threshold", 0.25)),
            max_detections=int(config.get("max_detections", 100)),
            class_names=config.get("class_names"),
        )
    
    elif backend_type == "mask_rcnn":
        from .mask_rcnn import MaskRCNNBackend
        return MaskRCNNBackend(
            device=config.get("device"),
            score_threshold=float(config.get("score_threshold", 0.25)),
            max_detections=int(config.get("max_detections", 100)),
        )
    
    else:  # hybrid (default)
        from .hybrid import HybridBackend
        food_classes = None
        if config.get("semantic_food_classes"):
            food_classes = set(config["semantic_food_classes"].values())
        return HybridBackend(
            device=config.get("device"),
            score_threshold=float(config.get("score_threshold", 0.25)),
            max_detections=int(config.get("max_detections", 100)),
            semantic_food_classes=food_classes,
        )
