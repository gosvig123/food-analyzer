"""Object detection utilities for locating food items in an image."""

from __future__ import annotations

from typing import List
import warnings

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import detection as detection_models
import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None
try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
except Exception:  # pragma: no cover - optional
    sam_model_registry = None
    SamPredictor = None

from .types import Detection


class FoodDetector:
    """Wrapper around a detection model with a heuristic fallback."""

    def __init__(
        self,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        device: str | None = None,
        backend: str = "torchvision_fasterrcnn",
        model_name: str | None = None,
        imgsz: int | None = None,
        retina_masks: bool = True,
        augment: bool = False,
        tta_imgsz: list[int] | None = None,
        refine_masks: bool = True,
        refine_method: str = "morphology",  # options: morphology, sam
        morph_kernel: int = 3,
        morph_iters: int = 1,
        sam_checkpoint: str | None = None,
        sam_model_type: str | None = None,
        fusion_method: str = "soft_nms",  # options: nms, soft_nms
        soft_nms_sigma: float = 0.5,
    ) -> None:
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = int(max(1, max_detections))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        self.categories: List[str] = ["food"]
        self.preprocess = transforms.Compose([transforms.ToTensor()])
        self.imgsz = imgsz
        self.retina_masks = bool(retina_masks)
        self.augment = bool(augment)
        self.tta_imgsz = tta_imgsz or []
        self.refine_masks = bool(refine_masks)
        self.refine_method = (refine_method or "morphology").lower()
        self.morph_kernel = int(max(1, morph_kernel))
        self.morph_iters = int(max(0, morph_iters))
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self._sam_predictor = None
        self.fusion_method = (fusion_method or "nms").lower()
        self.soft_nms_sigma = float(max(1e-6, soft_nms_sigma))

        try:
            if backend == "yolov8":
                try:
                    from ultralytics import YOLO  # type: ignore
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(f"ultralytics not available: {exc}")
                yolo_model = model_name or "yolov8n.pt"
                self.model = YOLO(yolo_model)
                names = getattr(self.model, "names", None)
                if isinstance(names, dict):
                    self.categories = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    self.categories = names
                else:
                    self.categories = ["food"]
            else:
                weights = detection_models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = detection_models.fasterrcnn_resnet50_fpn(weights=weights)
                self.model.to(self.device).eval()
                self.categories = weights.meta.get("categories", self.categories)
        except Exception as exc:  # pragma: no cover - best-effort offline fallback
            warnings.warn(f"Falling back to heuristic detector: {exc}")
            self.model = None
            self.backend = "heuristic"

    def _refine_polygon(self, image: Image.Image, polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        # SAM-based refinement
        if self.refine_method == "sam":
            try:
                predictor = self._ensure_sam()
                if predictor is None:
                    return polygon
                # Compute bbox from polygon
                xs = [float(x) for x, _ in polygon]; ys = [float(y) for _, y in polygon]
                x1, y1, x2, y2 = max(0.0, min(xs)), max(0.0, min(ys)), max(xs), max(ys)
                # Prepare image for predictor
                im = np.array(image.convert("RGB"))
                predictor.set_image(im)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array([x1, y1, x2, y2])[None, :],
                    multimask_output=False,
                )
                mask = masks[0].astype(np.uint8) * 255
                return self._mask_to_polygon(mask)
            except Exception:
                # Fallback to morphology if SAM fails
                pass
        # Morphological refinement
        if cv2 is None:
            return polygon
        w, h = image.size
        pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32).reshape((-1, 1, 2))
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
        if not self.sam_checkpoint or not self.sam_model_type:
            return None
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
            sam.to(self.device)
            self._sam_predictor = SamPredictor(sam)
            return self._sam_predictor
        except Exception:
            return None

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
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * max(1.0, peri), True)
        poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
        step = max(1, int(len(poly) / 300))
        return poly[::step]


    def __call__(self, image: Image.Image) -> List[Detection]:
        image = image.convert("RGB")

        if self.model is not None and self.backend == "yolov8":
            try:
                yargs = dict(
                    verbose=False,
                    conf=float(self.score_threshold),
                    iou=float(self.iou_threshold),
                    max_det=int(self.max_detections),
                    retina_masks=self.retina_masks,
                    augment=self.augment,
                )
                if self.imgsz:
                    yargs["imgsz"] = int(self.imgsz)
                results = self.model(image, **yargs)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                warnings.warn(f"YOLOv8 inference failed, falling back: {exc}")
                results = []
            def parse_results(rs) -> List[Detection]:
                parsed: List[Detection] = []
                for r in rs:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None:
                        continue
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        continue
                    masks = getattr(r, "masks", None)
                    masks_xy = getattr(masks, "xy", None) if masks is not None else None
                    for idx, ((left, top, right, bottom), score, cls_id) in enumerate(zip(xyxy, confs, clss)):
                        if float(score) < float(self.score_threshold):
                            continue
                        label = self.categories[cls_id] if 0 <= cls_id < len(self.categories) else "food"
                        mask_polygon = None
                        if masks_xy is not None and idx < len(masks_xy):
                            try:
                                poly = masks_xy[idx]
                                # Downsample very long polygons to keep JSON small
                                step = max(1, int(len(poly) / 300))
                                mask_polygon = [(float(x), float(y)) for x, y in poly[::step]]
                            except Exception:
                                mask_polygon = None
                        # Optional mask refinement
                        if mask_polygon and self.refine_masks:
                            try:
                                mask_polygon = self._refine_polygon(image, mask_polygon)
                            except Exception:
                                pass
                        parsed.append(
                            Detection(
                                box=(int(round(left)), int(round(top)), int(round(right)), int(round(bottom))),
                                confidence=float(score),
                                label=label,
                                mask_polygon=mask_polygon,
                            )
                        )
                return parsed

            width, height = image.size

            def iou_box(a: Detection, b: Detection) -> float:
                ax1, ay1, ax2, ay2 = a.box
                bx1, by1, bx2, by2 = b.box
                inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
                iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
                inter = iw * ih
                area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                union = area_a + area_b - inter
                return float(inter / union) if union > 0 else 0.0

            def nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
                out: List[Detection] = []
                for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
                    if all(iou_box(d, kept) < iou_thr for kept in out):
                        out.append(d)
                    if len(out) >= int(self.max_detections):
                        break
                return out

            def soft_nms(dets: List[Detection], iou_thr: float, sigma: float, score_thresh: float) -> List[Detection]:
                boxes = dets[:]
                keep: List[Detection] = []
                while boxes and len(keep) < int(self.max_detections):
                    boxes.sort(key=lambda x: x.confidence, reverse=True)
                    best = boxes.pop(0)
                    keep.append(best)
                    new_boxes: List[Detection] = []
                    for d in boxes:
                        iou = iou_box(best, d)
                        if iou > iou_thr:
                            decay = float(np.exp(-(iou * iou) / max(1e-6, sigma)))
                            new_score = float(d.confidence) * decay
                        else:
                            new_score = float(d.confidence)
                        if new_score >= float(score_thresh):
                            new_boxes.append(Detection(box=d.box, confidence=new_score, label=d.label, mask_polygon=d.mask_polygon))
                    boxes = new_boxes
                return keep

            if self.augment:
                # Manual TTA: multi-scale + flips (none, H, V, HV)
                all_dets: List[Detection] = []
                scales = self.tta_imgsz if self.tta_imgsz else ([self.imgsz] if self.imgsz else [None])
                flip_modes = ["none", "h", "v", "hv"]
                for scale in scales:
                    yargs_tta = dict(
                        verbose=False,
                        conf=float(self.score_threshold),
                        iou=float(self.iou_threshold),
                        max_det=int(self.max_detections),
                        retina_masks=self.retina_masks,
                        augment=False,
                    )
                    if scale:
                        yargs_tta["imgsz"] = int(scale)
                    for fm in flip_modes:
                        try:
                            img_in = image
                            if fm in ("h", "hv"):
                                img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
                            if fm in ("v", "hv"):
                                img_in = img_in.transpose(Image.FLIP_TOP_BOTTOM)
                            rs = self.model(img_in, **yargs_tta)
                            dets_raw: List[Detection] = parse_results(rs)
                            # Unflip coordinates back to original frame
                            for d in dets_raw:
                                l, t, r, b = d.box
                                new_l, new_r, new_t, new_b = l, r, t, b
                                new_poly = d.mask_polygon
                                if fm in ("h", "hv"):
                                    new_l, new_r = width - r, width - l
                                    if new_poly:
                                        try:
                                            new_poly = [(float(width - x), float(y)) for x, y in new_poly]
                                        except Exception:
                                            new_poly = None
                                if fm in ("v", "hv"):
                                    new_t, new_b = height - b, height - t
                                    if new_poly:
                                        try:
                                            new_poly = [(float(x), float(height - y)) for x, y in new_poly]
                                        except Exception:
                                            new_poly = None
                                # Optional mask refinement
                                if new_poly and self.refine_masks:
                                    try:
                                        new_poly = self._refine_polygon(image, new_poly)
                                    except Exception:
                                        pass
                                all_dets.append(
                                    Detection(box=(int(new_l), int(new_t), int(new_r), int(new_b)),
                                              confidence=d.confidence,
                                              label=d.label,
                                              mask_polygon=new_poly)
                                )
                        except Exception:
                            continue
                if all_dets:
                    if self.fusion_method == "soft_nms":
                        detections = soft_nms(all_dets, float(self.iou_threshold), float(self.soft_nms_sigma), float(self.score_threshold))
                    else:
                        detections = nms(all_dets, float(self.iou_threshold))
                else:
                    detections = []
            else:
                detections: List[Detection] = parse_results(results)

                # Adaptive per-image thresholding: retry once if none or too many detections
                if (len(detections) == 0) or (len(detections) > int(self.max_detections)):
                    try:
                        adj_conf = (
                            max(0.1, float(self.score_threshold) - 0.2)
                            if len(detections) == 0
                            else min(0.9, float(self.score_threshold) + 0.2)
                        )
                        yargs = dict(
                            verbose=False,
                            conf=float(adj_conf),
                            iou=float(self.iou_threshold),
                            max_det=int(self.max_detections),
                            retina_masks=self.retina_masks,
                            augment=False,
                        )
                        if self.imgsz:
                            yargs["imgsz"] = int(self.imgsz)
                        retry = self.model(image, **yargs)
                        detections = parse_results(retry)
                    except Exception:
                        pass

            if detections:
                return detections

        tensor = self.preprocess(image)
        if self.model is not None and self.backend != "yolov8":
            with torch.no_grad():
                outputs = self.model([tensor.to(self.device)])[0]

            def build_from_outputs(threshold: float) -> List[Detection]:
                built: List[Detection] = []
                for score, label_idx, box in zip(
                    outputs.get("scores", []).tolist(),
                    outputs.get("labels", []).tolist(),
                    outputs.get("boxes", []).tolist(),
                ):
                    if score < threshold:
                        continue
                    left, top, right, bottom = (int(round(v)) for v in box)
                    category = self.categories[label_idx] if 0 <= label_idx < len(self.categories) else "food"
                    built.append(
                        Detection(
                            box=(left, top, right, bottom),
                            confidence=float(score),
                            label=category,
                        )
                    )
                return built

            detections = build_from_outputs(self.score_threshold)
            max_detections = 20
            if (len(detections) == 0) or (len(detections) > max_detections):
                alt_th = (
                    max(0.1, float(self.score_threshold) - 0.2)
                    if len(detections) == 0
                    else min(0.9, float(self.score_threshold) + 0.2)
                )
                detections = build_from_outputs(alt_th)

            if detections:
                return detections

        width, height = image.size
        return [Detection(box=(0, 0, width, height), confidence=1.0, label="food")]


__all__ = ["FoodDetector"]
