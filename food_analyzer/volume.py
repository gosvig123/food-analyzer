"""Heuristics for estimating serving sizes from detections."""

from __future__ import annotations

from typing import Tuple

import torch


class VolumeEstimator:
    """Estimates food mass from bounding-box area relative to the frame."""

    def __init__(self, grams_for_full_plate: float = 300.0) -> None:
        self.grams_for_full_plate = grams_for_full_plate

    def __call__(
        self,
        box: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        depth_map: torch.Tensor | None = None,
        depth_mean: float | None = None,
    ) -> float:
        left, top, right, bottom = box
        width, height = image_size
        frame_area = max(width * height, 1)
        box_area = max((right - left) * (bottom - top), 1)
        ratio = min(box_area / frame_area, 1.0)

        depth_scale = 1.0
        if depth_map is not None:
            h, w = depth_map.shape[-2:]
            left_i = max(int(left), 0)
            right_i = min(int(right), w)
            top_i = max(int(top), 0)
            bottom_i = min(int(bottom), h)
            region = depth_map[top_i:bottom_i, left_i:right_i]
            if region.numel() > 0:
                global_mean = depth_mean or float(depth_map.mean().item())
                if global_mean > 0:
                    region_mean = float(region.mean().item())
                    depth_scale = max(0.5, min(region_mean / global_mean, 2.0))

        return round(self.grams_for_full_plate * ratio * depth_scale, 1)


__all__ = ["VolumeEstimator"]
