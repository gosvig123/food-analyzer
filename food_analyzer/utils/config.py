"""Lightweight configuration loader for the food-analyzer project.

- Prefers JSON config to avoid extra dependencies
- Optionally supports YAML if PyYAML is available
- Provides sane defaults aligned with the current minimal pipeline
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

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


@dataclass
class IngredientConfig:
    """Configuration for ingredient label fetching."""
    
    apis: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    search_categories: List[str] = field(default_factory=list)
    food_keywords: Dict[str, List[str]] = field(default_factory=dict)
    cleaning_patterns: List[str] = field(default_factory=list)
    generic_terms: List[str] = field(default_factory=list)
    non_ingredient_terms: List[str] = field(default_factory=list)
    cache_maxsize: int = 1
    cache_file: str = "ingredient_cache.json"
    fallback_ingredients: List[str] = field(default_factory=list)


def load_ingredient_config(path: Optional[str | Path] = None) -> IngredientConfig:
    """Load ingredient configuration from ingredient_config.json."""
    if path is None:
        path = Path("ingredient_config.json")
    else:
        path = Path(path)
    
    if not path.exists():
        return IngredientConfig()
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        return IngredientConfig(
            apis=data.get("apis", {}),
            models=data.get("models", {}),
            search_categories=data.get("search_categories", []),
            food_keywords=data.get("food_keywords", {}),
            cleaning_patterns=data.get("cleaning_patterns", []),
            generic_terms=data.get("generic_terms", []),
            non_ingredient_terms=data.get("non_ingredient_terms", []),
            cache_maxsize=data.get("cache_maxsize", 1),
            cache_file=data.get("cache_file", "ingredient_cache.json"),
            fallback_ingredients=data.get("fallback_ingredients", []),
        )
    except Exception as exc:
        warnings.warn(f"Failed to load ingredient config from {path}: {exc}")
        return IngredientConfig()
