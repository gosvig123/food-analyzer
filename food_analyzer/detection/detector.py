"""Object detection utilities for locating food items in an image."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import detection as detection_models
from torchvision.models import segmentation as segmentation_models

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None
try:
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore
except Exception:  # pragma: no cover - optional
    sam_model_registry = None
    SamPredictor = None

from ..core.types import Detection


class FoodDetector:
    """High-recall food detection with hybrid segmentation approach."""

    def __init__(
        self,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        device: str | None = None,
        backend: str = "mask_rcnn_deeplabv3",
        model_name: str | None = None,
        imgsz: int | None = None,
        retina_masks: bool = True,
        augment: bool = False,
        tta_imgsz: list[int] | None = None,
        refine_masks: bool = True,
        refine_method: str = "sam",
        morph_kernel: int = 5,
        morph_iters: int = 2,
        sam_checkpoint: str | None = None,
        sam_model_type: str | None = None,
        fusion_method: str = "soft_nms",
        soft_nms_sigma: float = 0.5,
        semantic_food_classes: dict[str, int] | None = None,
    ) -> None:
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = int(max(1, max_detections))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        self.semantic_model = None
        self.categories: List[str] = ["food"]
        self.preprocess = transforms.Compose([transforms.ToTensor()])
        self.semantic_preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
        self.imgsz = imgsz
        self.retina_masks = bool(retina_masks)
        self.augment = bool(augment)
        self.tta_imgsz = tta_imgsz or []
        self.refine_masks = bool(refine_masks)
        self.refine_method = (refine_method or "sam").lower()
        self.morph_kernel = int(max(1, morph_kernel))
        self.morph_iters = int(max(0, morph_iters))
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self._sam_predictor = None
        self.fusion_method = (fusion_method or "soft_nms").lower()
        self.soft_nms_sigma = float(max(1e-6, soft_nms_sigma))
        # Configurable semantic food classes (COCO class IDs for DeepLabV3+)
        # Default: common food classes from COCO
        self.semantic_food_classes: set[int] = set(
            (semantic_food_classes or {
                "apple": 47, "orange": 48, "banana": 49, "carrot": 50,
                "hot_dog": 51, "pizza": 52, "donut": 53, "cake": 54,
            }).values()
        )
        self._has_gpu = self.device.startswith("cuda") and torch.cuda.is_available()

        if not self._has_gpu:
            if self.augment or self.tta_imgsz:
                self.augment = False
                self.tta_imgsz = []
            if self.refine_method == "sam":
                # Fall back to lightweight morphological refinement when GPUs are unavailable
                self.refine_method = "morph"
            if self.max_detections > 75:
                self.max_detections = 75

        self._sam_checkpoint_path: Path | None = None
        if self.sam_checkpoint:
            candidate = Path(self.sam_checkpoint)
            if candidate.exists():
                self._sam_checkpoint_path = candidate
            else:
                warnings.warn(
                    f"SAM checkpoint not found at {candidate}. Skipping SAM-based refinement."
                )
                self.sam_checkpoint = None
                if self.refine_method == "sam":
                    self.refine_method = "morph"

        try:
            if backend == "mask_rcnn_deeplabv3":
                # Load Mask R-CNN for instance segmentation
                mask_weights = detection_models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = detection_models.maskrcnn_resnet50_fpn_v2(
                    weights=mask_weights
                )
                self.model.to(self.device).eval()

                # Load DeepLabV3+ for semantic segmentation
                if not self._has_gpu:
                    self.semantic_model = None
                else:
                    semantic_weights = (
                        segmentation_models.DeepLabV3_ResNet50_Weights.DEFAULT
                    )
                    self.semantic_model = segmentation_models.deeplabv3_resnet50(
                        weights=semantic_weights
                    )
                    self.semantic_model.to(self.device).eval()

                self.categories = mask_weights.meta.get("categories", self.categories)
            elif backend == "mask_rcnn":
                weights = detection_models.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = detection_models.maskrcnn_resnet50_fpn_v2(weights=weights)
                self.model.to(self.device).eval()
                self.categories = weights.meta.get("categories", self.categories)
            else:
                weights = detection_models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = detection_models.fasterrcnn_resnet50_fpn(weights=weights)
                self.model.to(self.device).eval()
                self.categories = weights.meta.get("categories", self.categories)
        except Exception as exc:  # pragma: no cover - best-effort offline fallback
            warnings.warn(f"Model loading failed: {exc}")
            self.model = None

    def _refine_polygon(
        self, image: Image.Image, polygon: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        # SAM-based refinement for optimal accuracy
        if self.refine_method == "sam":
            try:
                predictor = self._ensure_sam()
                if predictor is None:
                    return polygon
                # Compute bbox from polygon
                xs = [float(x) for x, _ in polygon]
                ys = [float(y) for _, y in polygon]
                x1, y1, x2, y2 = max(0.0, min(xs)), max(0.0, min(ys)), max(xs), max(ys)
                # Prepare image for predictor
                im = np.array(image.convert("RGB"))
                predictor.set_image(im)
                # Use multimask for better accuracy
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([x1, y1, x2, y2])[None, :],
                    multimask_output=True,
                )
                # Select best mask by score
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.uint8) * 255
                return self._mask_to_polygon(mask)
            except Exception:
                pass
        # Morphological refinement fallback
        if cv2 is None:
            return polygon
        w, h = image.size
        pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32).reshape(
            (-1, 1, 2)
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        k = max(1, int(self.morph_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        iters = max(0, int(self.morph_iters))
        if iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        return self._mask_to_polygon(mask)

    def _ensure_sam(self):
        if getattr(self, "_sam_predictor", None) is not None:
            return self._sam_predictor
        if SamPredictor is None or sam_model_registry is None:
            return None
        if not self.sam_model_type:
            return None
        checkpoint = self._sam_checkpoint_path
        if checkpoint is None or not checkpoint.exists():
            return None
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=str(checkpoint))
            sam.to(self.device)
            self._sam_predictor = SamPredictor(sam)
            return self._sam_predictor
        except Exception:
            return None

    def _semantic_segmentation_to_masks(self, image: Image.Image) -> list[np.ndarray]:
        """Use DeepLabV3+ to generate semantic segmentation masks for food regions."""
        if self.semantic_model is None:
            return []

        # Preprocess for semantic segmentation
        img_tensor = self.preprocess(image).unsqueeze(0)
        img_tensor = self.semantic_preprocess(img_tensor).to(self.device)

        with torch.inference_mode():
            output = self.semantic_model(img_tensor)["out"]
            # Get class predictions
            predictions = torch.softmax(output, dim=1)
            pred_classes = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()

        # Use configurable food classes (set via constructor or config)
        masks = []
        for food_class in self.semantic_food_classes:
            mask = (pred_classes == food_class).astype(np.uint8) * 255
            if mask.sum() > 100:  # Filter out very small segments
                masks.append(mask)

        return masks

    def _mask_to_polygon(self, mask: np.ndarray) -> list[tuple[float, float]]:
        if cv2 is None:
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
        # High-precision polygon extraction with adaptive approximation
        peri = cv2.arcLength(cnt, True)
        # Use adaptive epsilon based on contour complexity
        epsilon = 0.001 * max(1.0, peri) if len(cnt) > 500 else 0.002 * max(1.0, peri)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
        # Preserve more detail for smaller objects
        max_points = min(300, max(50, int(peri / 10)))
        step = max(1, int(len(poly) / max_points))
        return poly[::step]

    def _inference_mask_rcnn_deeplabv3(self, image: Image.Image) -> List[Detection]:
        """Inference using Mask R-CNN + DeepLabV3+ hybrid approach for better recall."""
        # Multi-scale inference for better detection of various food sizes
        scales = [1.0]
        if self.tta_imgsz and self.augment:
            original_size = min(image.size)
            for target_size in self.tta_imgsz:
                if target_size != original_size:
                    scales.append(target_size / original_size)

        all_instance_detections = []
        all_instance_masks = []

        for scale in scales:
            if scale != 1.0:
                scaled_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                scaled_image = image.resize(scaled_size, Image.Resampling.LANCZOS)
            else:
                scaled_image = image

            # Instance segmentation with Mask R-CNN
            tensor = self.preprocess(scaled_image).to(self.device)

            with torch.inference_mode():
                outputs = self.model([tensor])[0]

            # Extract instance masks for this scale
            scale_instance_masks = []
            scale_instance_detections = []

            for score, label_idx, box, mask in zip(
                outputs.get("scores", []).cpu().numpy(),
                outputs.get("labels", []).cpu().numpy(),
                outputs.get("boxes", []).cpu().numpy(),
                outputs.get("masks", []).cpu().numpy(),
            ):
                if score < self.score_threshold:
                    continue

                # Scale coordinates back to original image size
                if scale != 1.0:
                    left, top, right, bottom = [int(round(v / scale)) for v in box]
                else:
                    left, top, right, bottom = (int(round(v)) for v in box)

                category = (
                    self.categories[label_idx]
                    if 0 <= label_idx < len(self.categories)
                    else "food"
                )

                # Convert mask to binary and scale back if needed
                mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
                if scale != 1.0:
                    mask_binary = np.array(
                        Image.fromarray(mask_binary).resize(
                            image.size, Image.Resampling.NEAREST
                        )
                    )

                scale_instance_masks.append(mask_binary)

                mask_polygon = self._mask_to_polygon(mask_binary)
                if mask_polygon and self.refine_masks:
                    try:
                        mask_polygon = self._refine_polygon(image, mask_polygon)
                    except Exception:
                        pass

                scale_instance_detections.append(
                    Detection(
                        box=(left, top, right, bottom),
                        confidence=float(score),
                        label=category,
                        mask_polygon=mask_polygon,
                    )
                )

            all_instance_detections.extend(scale_instance_detections)
            all_instance_masks.extend(scale_instance_masks)

        # Merge multi-scale detections
        instance_detections = all_instance_detections
        instance_masks = all_instance_masks

        # Semantic segmentation with DeepLabV3+
        semantic_masks = self._semantic_segmentation_to_masks(image)

        # Process all semantic masks without overlap filtering
        additional_detections = []
        for sem_mask in semantic_masks:
            # Create detection from semantic mask
            if cv2 is not None:
                contours, _ = cv2.findContours(
                    sem_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    mask_polygon = self._mask_to_polygon(sem_mask)
                    if mask_polygon and self.refine_masks:
                        try:
                            mask_polygon = self._refine_polygon(image, mask_polygon)
                        except Exception:
                            pass

                    # Enhanced confidence scoring based on mask quality and size
                    mask_area = cv2.contourArea(largest_contour)
                    total_area = sem_mask.shape[0] * sem_mask.shape[1]
                    area_ratio = mask_area / total_area

                    # Improved confidence calculation with size and shape analysis
                    bbox_area = w * h
                    aspect_ratio = max(w, h) / max(min(w, h), 1)

                    # Base confidence from area ratio
                    base_confidence = 0.5 + (area_ratio * 0.3)

                    # Penalty for extreme aspect ratios (likely non-food objects)
                    if aspect_ratio > 3.0:
                        base_confidence *= 0.7

                    # Bonus for reasonable food-sized objects
                    if 500 < bbox_area < 50000:
                        base_confidence *= 1.1
                    elif bbox_area < 200:  # Too small, likely noise
                        base_confidence *= 0.5

                    confidence = min(0.8, max(0.3, base_confidence))

                    additional_detections.append(
                        Detection(
                            box=(x, y, x + w, y + h),
                            confidence=confidence,
                            label="food",
                            mask_polygon=mask_polygon,
                        )
                    )

        # Combine all detections - allow full overlap for ingredients
        all_detections = instance_detections + additional_detections

        # Sort by confidence and return top detections (no overlap filtering)
        all_detections.sort(key=lambda x: x.confidence, reverse=True)
        return all_detections[: self.max_detections]

    def __call__(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")

        # Handle Mask R-CNN + DeepLabV3+ hybrid backend for maximum recall
        if self.model is not None and self.backend == "mask_rcnn_deeplabv3":
            return self._inference_mask_rcnn_deeplabv3(image)

        # Handle basic Mask R-CNN without overlap filtering
        if self.model is not None and self.backend == "mask_rcnn":
            tensor = self.preprocess(image).to(self.device)
            with torch.inference_mode():
                outputs = self.model([tensor])[0]

            detections = []
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
                    self.categories[label_idx]
                    if 0 <= label_idx < len(self.categories)
                    else "food"
                )

                mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
                mask_polygon = self._mask_to_polygon(mask_binary)
                if mask_polygon and self.refine_masks:
                    try:
                        mask_polygon = self._refine_polygon(image, mask_polygon)
                    except Exception:
                        pass

                detections.append(
                    Detection(
                        box=(left, top, right, bottom),
                        confidence=float(score),
                        label=category,
                        mask_polygon=mask_polygon,
                    )
                )

            # Sort by confidence and return without overlap filtering
            detections.sort(key=lambda x: x.confidence, reverse=True)
            return detections[: self.max_detections]

        # No fallback - fail gracefully if model not available
        return []


__all__ = ["FoodDetector"]
