"""Lightweight configuration loader for the food-analyzer project.

- Prefers JSON config to avoid extra dependencies
- Optionally supports YAML if PyYAML is available
- Provides sane defaults aligned with the current minimal pipeline
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "detector": {
        "backend": "torchvision_fasterrcnn",  # or 'yolov8'
        "model_name": "yolov8n.pt",  # used when backend == 'yolov8'
        "score_threshold": 0.5,
        "iou_threshold": 0.45,
        "max_detections": 100,
        "device": None,
    },
    "classifier": {
        "backend": "efficientnet_b0",  # or 'efficientnet_b4'
        "device": None,
        "dynamic_labels_source": "usda",  # API fallback: "usda", "openfoodfacts", or "basic"
        "intelligent_labels_method": "hybrid",  # "imagenet", "clip", "nutrition", or "hybrid"
    },
    "volume": {
        "grams_for_full_plate": 300.0,
    },
    "nutrition": {
        # Relative to project root by default
        "table_path": "food_analyzer/nutrition_defaults.json",
    },
    "depth": {
        "enabled": True,
    },
    "pipeline": {
        "use_detector_labels": False,
        "maximize_recall": False,
    },
    "io": {
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".gif"],
        "results_dir": "results",
        "save_overlays": False,
        "save_crops": False,
        "save_masks": False,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        warnings.warn(
            f"PyYAML not installed; cannot parse YAML config '{path}'. Using defaults. ({exc})"
        )
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration from a file (JSON preferred; YAML optional) and merge with defaults.

    Lookup order:
    1) Explicit path if provided
    2) ./config.json in project root if present
    3) ./config.yaml or ./config.yml if present
    4) Defaults
    """
    merged = json.loads(json.dumps(DEFAULTS))  # deep copy via JSON round-trip

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    # Project root assumed as CWD
    candidates.extend([Path("config.json"), Path("config.yaml"), Path("config.yml")])

    chosen: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if not chosen:
        return merged

    try:
        if chosen.suffix.lower() == ".json":
            data = _load_json(chosen)
        elif chosen.suffix.lower() in {".yaml", ".yml"}:
            data = _load_yaml(chosen)
        else:
            warnings.warn(f"Unsupported config format: {chosen.suffix}. Using defaults.")
            data = {}
    except Exception as exc:
        warnings.warn(f"Failed to load config from {chosen}: {exc}. Using defaults.")
        data = {}

    if isinstance(data, dict):
        _deep_update(merged, data)
    else:
        warnings.warn(f"Config at {chosen} is not a mapping. Using defaults.")

    return merged


def resolve_path_relative_to_project(path_str: str | None) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    # Try relative to project root (CWD)
    cwd_p = Path.cwd() / p
    if cwd_p.exists():
        return cwd_p
    # Try relative to package directory
    pkg_p = Path(__file__).resolve().parent / p
    if pkg_p.exists():
        return pkg_p
    return p  # return original Path even if not existing, caller may handle
