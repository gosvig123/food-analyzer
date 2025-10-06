#!/usr/bin/env python3
"""Generate HTML evaluation report from existing ground_truth_evaluation.json"""

import json
import sys
from pathlib import Path

from food_analyzer.io.results_writer import ResultsWriter


def main():
    results_dir = Path("results")
    eval_path = results_dir / "ground_truth_evaluation.json"
    
    if not eval_path.exists():
        print(f"Error: {eval_path} not found")
        print("Run inference first to generate evaluation data")
        return 1
    
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    
    validation_results = eval_data.get("detailed_results", [])
    
    writer = ResultsWriter(
        results_dir=results_dir,
        save_overlays=False,
        save_crops=False,
        save_masks=False,
    )
    
    writer._generate_html_report(validation_results, eval_data)
    
    html_path = results_dir / "evaluation_report.html"
    print(f"\n✓ HTML report generated: {html_path}")
    print(f"\nOpen in browser:")
    print(f"  open {html_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
