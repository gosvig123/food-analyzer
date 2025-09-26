"""Utilities for managing dynamic ingredient labels and their aliases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple


def _normalize_token(value: str) -> str:
    """Normalize label tokens for consistent matching."""
    if not value:
        return ""
    # Replace separators with spaces, collapse whitespace, and lowercase
    collapsed = value.replace("-", " ").replace("_", " ")
    return " ".join(collapsed.strip().lower().split())


def _generate_aliases(label: str) -> set[str]:
    """Generate common alias spellings for a label dynamically."""
    base = _normalize_token(label)
    if not base:
        return set()

    aliases: set[str] = {base}

    tokens = base.split()
    if tokens:
        collapsed = "".join(tokens)
        aliases.add(collapsed)
        aliases.add("_".join(tokens))
        aliases.add("-".join(tokens))

    if base.endswith("ies"):
        aliases.add(base[:-3] + "y")
    if base.endswith("s") and not base.endswith("ss"):
        aliases.add(base[:-1])
    else:
        aliases.add(base + "s")

    if base.endswith("y"):
        aliases.add(base[:-1] + "ies")

    # Normalize again to ensure consistent formatting
    return {alias for alias in (_normalize_token(a) for a in aliases) if alias}


@dataclass
class LabelNormalizer:
    """Maps noisy label inputs to canonical dynamic labels."""

    alias_to_canonical: Dict[str, str]
    canonical_to_display: Dict[str, str]

    @classmethod
    def from_labels(
        cls,
        labels: Iterable[str],
        extra_synonyms: Optional[Mapping[str, str]] = None,
    ) -> "LabelNormalizer":
        alias_to_canonical: Dict[str, str] = {}
        canonical_to_display: Dict[str, str] = {}

        for label in labels:
            display = label.strip()
            canonical = _normalize_token(display)
            if not canonical:
                continue

            canonical_to_display.setdefault(canonical, display)

            for alias in _generate_aliases(display):
                alias_to_canonical.setdefault(alias, canonical)
            # Ensure canonical form resolves to itself
            alias_to_canonical.setdefault(canonical, canonical)

        if extra_synonyms:
            for alias, target in extra_synonyms.items():
                alias_key = _normalize_token(alias)
                target_key = _normalize_token(target)
                if not alias_key or not target_key:
                    continue
                # Prefer an existing canonical mapping for the target
                canonical_target = alias_to_canonical.get(target_key)
                if not canonical_target and target_key in canonical_to_display:
                    canonical_target = target_key
                if canonical_target:
                    alias_to_canonical[alias_key] = canonical_target

        return cls(
            alias_to_canonical=alias_to_canonical,
            canonical_to_display=canonical_to_display,
        )

    def normalize(self, value: Optional[str]) -> Optional[str]:
        """Return the canonical label for the given value if known."""
        if not value:
            return None
        return self.alias_to_canonical.get(_normalize_token(value))

    def display(self, canonical: str) -> str:
        """Return the preferred display value for a canonical label."""
        return self.canonical_to_display.get(canonical, canonical)


def load_synonym_map(path: Path = Path("dynamic_synonyms.json")) -> Dict[str, str]:
    """Load optional user-provided synonyms that map to dynamic labels."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                # Ensure keys and values are strings before returning
                return {
                    str(key): str(value)
                    for key, value in data.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
    except Exception:
        pass
    return {}


def align_ground_truth_with_labels(
    ground_truth: Mapping[str, Iterable[str]],
    normalizer: LabelNormalizer,
) -> Tuple[Dict[str, set[str]], Dict[str, list[str]]]:
    """Align ground truth ingredient labels with canonical dynamic labels.

    Returns a tuple of (aligned_ground_truth, unmatched_items) where the aligned
    mapping uses canonical labels and unmatched_items records any ingredients
    that could not be mapped.
    """
    aligned: Dict[str, set[str]] = {}
    unmatched: Dict[str, list[str]] = {}

    for plate_type, ingredients in ground_truth.items():
        canonical_set: set[str] = set()
        missing: list[str] = []

        for ingredient in ingredients:
            canonical = normalizer.normalize(ingredient)
            if canonical:
                canonical_set.add(canonical)
            else:
                missing.append(str(ingredient))

        aligned[plate_type] = canonical_set
        if missing:
            unmatched[plate_type] = missing

    return aligned, unmatched
