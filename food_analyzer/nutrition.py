"""Nutrition lookup helpers for converting labels into macro estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_default_nutrition_table() -> Dict[str, Dict[str, float]]:
    """Load the default nutrition table shipped with the package."""

    path = Path(__file__).with_name("nutrition_defaults.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class NutritionLookup:
    """Looks up macro nutrients using an external table (JSON). No defaults or fallbacks."""

    def __init__(
        self,
        table: Dict[str, Dict[str, float]] | None = None,
        table_path: str | Path | None = None,
    ) -> None:
        if table is not None:
            self.table = table
        else:
            self.table = {}
            if table_path:
                candidate = Path(table_path)
                try:
                    with candidate.open("r", encoding="utf-8") as handle:
                        self.table = json.load(handle)
                except Exception:
                    self.table = {}

    def __call__(self, label: str, grams: float) -> Dict[str, float]:
        record = self.table.get(label)
        if not record:
            return {}
        multiplier = grams / 100.0
        return {key: round(value * multiplier, 2) for key, value in record.items()}


__all__ = ["load_default_nutrition_table", "NutritionLookup"]
