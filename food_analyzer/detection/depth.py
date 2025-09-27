"""Depth estimation helpers used to scale serving size by perceived volume."""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DepthEstimator:
    """Predicts per-pixel relative depth using a MiDaS backbone."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        try:
            # Suppress the misleading "Loading weights: None" message from MiDaS
            import io
            import sys
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                self.model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                self.model.to(self.device).eval()
                transforms = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                )
                self.transform = transforms.small_transform
        except Exception as exc:  # pragma: no cover - best-effort offline fallback
            warnings.warn(f"Depth estimator unavailable, using flat depth map: {exc}")
            self.model = None
            self.transform = None

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        if self.model is None or self.transform is None:
            width, height = image.size
            return torch.ones((height, width))

        array = np.asarray(image)
        input_batch = self.transform(array).to(self.device)
        with torch.inference_mode():
            prediction = self.model(input_batch)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(0)
            prediction = F.interpolate(
                prediction,
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
        return prediction.squeeze().cpu()


__all__ = ["DepthEstimator"]
