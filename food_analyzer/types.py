"""Shared dataclasses and type aliases used across food analyzer components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

from PIL import Image


ImageInput = Union[str, Path, Image.Image]


@dataclass
class Detection:
    """A single detection bounding box with confidence, label, and optional mask polygon."""

    box: Tuple[int, int, int, int]
    confidence: float
    label: str
    mask_polygon: list[tuple[float, float]] | None = None


@dataclass
class AnalyzedFood:
    """Enriched analysis output after running the full pipeline."""

    label: str
    confidence: float
    box: Tuple[int, int, int, int]
    portion_grams: float
    nutrition: Dict[str, float]
    mask_polygon: list[tuple[float, float]] | None = None
