"""Dynamic ingredient label fetcher using food APIs."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

from ..data.ingredient_config import IngredientConfig, load_ingredient_config


class IngredientLabelFetcher:
    """Fetches ingredient labels dynamically from food APIs with local caching."""

    def __init__(
        self,
        cache_file: str | Path | None = None,
        config: Optional[IngredientConfig] = None,
    ):
        self.config = config or load_ingredient_config()
        self.cache_file = (
            Path(cache_file) if cache_file else Path(self.config.cache_file)
        )
        self._cache: dict[str, List[str]] = self._load_cache()

    def _load_cache(self) -> dict[str, List[str]]:
        """Load cached ingredient lists from file."""
        if not self.cache_file.exists():
            return {}
        try:
            with self.cache_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            warnings.warn(f"Failed to load ingredient cache: {exc}")
            return {}

    def _save_cache(self) -> None:
        """Save ingredient cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to save ingredient cache: {exc}")

    @lru_cache(maxsize=None)  # Will be set dynamically
    def get_ingredient_labels(self, source: str = "usda") -> List[str]:
        # Set cache size from config
        self.get_ingredient_labels.__wrapped__.__defaults__ = (
            self.config.cache_maxsize,
        )
        """Get ingredient labels from specified source with caching."""
        cache_key = f"{source}_ingredients"

        # Return cached results if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch from API based on source
        try:
            if source == "usda":
                labels = self._fetch_usda_ingredients()
            elif source == "openfoodfacts":
                labels = self._fetch_openfoodfacts_ingredients()
            else:
                # Fallback to basic ingredient list
                labels = self._get_basic_ingredients()

            # Cache the results
            self._cache[cache_key] = labels
            self._save_cache()
            return labels

        except Exception as exc:
            warnings.warn(f"Failed to fetch ingredients from {source}: {exc}")
            # Return cached fallback or basic ingredients
            return self._cache.get(cache_key, self.config.fallback_ingredients)

    def _fetch_usda_ingredients(self) -> List[str]:
        """Fetch ingredient labels from USDA FoodData Central API."""
        api_config = self.config.apis.get("usda")
        if not api_config:
            return self.config.fallback_ingredients

        categories = self.config.search_categories

        all_ingredients: Set[str] = set()

        for category in categories:
            try:
                # Build query parameters from config
                params = {
                    **api_config.params,
                    "query": category,
                    "pageSize": api_config.page_size,
                }

                # Encode parameters
                query_string = urllib.parse.urlencode(params, doseq=True)
                url = f"{api_config.base_url}?{query_string}"

                # Make request
                req = urllib.request.Request(url)
                for header, value in api_config.headers.items():
                    req.add_header(header, value)

                with urllib.request.urlopen(
                    req, timeout=api_config.timeout
                ) as response:
                    data = json.loads(response.read().decode())

                # Extract food descriptions
                foods = data.get("foods", [])
                for food in foods:
                    description = food.get("description", "").lower().strip()
                    if (
                        description and len(description.split()) <= 3
                    ):  # Keep simple ingredients
                        # Clean up the description
                        description = (
                            description.replace(",", "").replace("raw", "").strip()
                        )
                        if description and len(description) > 2:
                            all_ingredients.add(description)

            except Exception as exc:
                warnings.warn(f"Failed to fetch {category} from USDA API: {exc}")
                continue

        # Convert to sorted list and limit size
        ingredients = sorted(list(all_ingredients))[: api_config.max_results]
        return ingredients if ingredients else self.config.fallback_ingredients

    def _fetch_openfoodfacts_ingredients(self) -> List[str]:
        """Fetch ingredient labels from Open Food Facts API."""
        api_config = self.config.apis.get("openfoodfacts")
        if not api_config:
            return self.config.fallback_ingredients

        try:
            # Build URL with parameters
            query_string = urllib.parse.urlencode(api_config.params)
            url = f"{api_config.base_url}?{query_string}"

            req = urllib.request.Request(url)
            for header, value in api_config.headers.items():
                req.add_header(header, value)

            with urllib.request.urlopen(req, timeout=api_config.timeout) as response:
                data = json.loads(response.read().decode())

            # Extract ingredient names
            ingredients = []
            tags = data.get("tags", [])

            for tag in tags[: api_config.max_results]:
                name = tag.get("name", "").strip().lower()
                if name and len(name.split()) <= 2:  # Keep simple ingredients
                    ingredients.append(name)

            return ingredients if ingredients else self.config.fallback_ingredients

        except Exception as exc:
            warnings.warn(f"Failed to fetch from Open Food Facts: {exc}")
            return self.config.fallback_ingredients

    def _get_basic_ingredients(self) -> List[str]:
        """Return a basic set of common ingredients as fallback."""
        return self.config.fallback_ingredients


def get_dynamic_ingredient_labels(
    source: str = "usda",
    cache_file: str | Path | None = None,
    config: IngredientConfig | None = None,
) -> list[str]:
    """
    Convenience function to get ingredient labels dynamically.

    Args:
        source: API source ("usda", "openfoodfacts", or "basic")
        cache_file: Path to cache file (optional)
        config: Configuration object (optional, will load default if None)

    Returns:
        List of ingredient label strings
    """
    fetcher = IngredientLabelFetcher(cache_file, config)
    return fetcher.get_ingredient_labels(source)


__all__ = ["IngredientLabelFetcher", "get_dynamic_ingredient_labels"]
