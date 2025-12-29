"""Compare detection results across multiple runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def compare_results(dirs: List[str], out_path: str | None = None) -> None:
    """Compare detection counts across result folders.
    
    Args:
        dirs: List of result directory paths to compare
        out_path: Optional CSV output path
    """
    paths = [Path(d) for d in dirs]
    
    # Map: image_stem -> {dir_name: count}
    per_image: Dict[str, Dict[str, int]] = {}
    dir_names = [p.name for p in paths]
    
    for p, name in zip(paths, dir_names):
        for jf in sorted(p.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue

            stem_source = data.get("image") if isinstance(data, dict) else None
            stem = Path(stem_source or jf).stem

            aggregates: List[dict] = []
            if isinstance(data, dict):
                agg = data.get("aggregated", [])
                if isinstance(agg, list):
                    aggregates = agg

            if not aggregates and isinstance(data, list):
                aggregates = data

            total_count = 0
            for entry in aggregates:
                try:
                    total_count += int(entry.get("count", 0))
                except Exception:
                    continue

            per_image.setdefault(stem, {})[name] = total_count
    
    # Print simple table
    headers = ["image"] + dir_names
    print("\n=== Comparison: detections per image ===")
    print(" | ".join(h.ljust(20) for h in headers))
    print("-+-".join(["-" * 20 for _ in headers]))
    
    rows: List[List[str]] = []
    for stem in sorted(per_image.keys()):
        row = [stem] + [str(per_image[stem].get(name, 0)) for name in dir_names]
        rows.append(row)
        print(" | ".join([stem.ljust(20)] + [c.ljust(20) for c in row[1:]]))
    
    # Optional CSV output
    if out_path:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(r)
        print(f"\nSaved CSV: {out_path}")
