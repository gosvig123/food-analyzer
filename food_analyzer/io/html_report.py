"""HTML report generator for ground truth evaluation."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List


class HTMLReportGenerator:
    """Generates interactive HTML reports for ground truth evaluation."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)

    def generate(self, validation_results: List[Dict], evaluation_report: Dict) -> str:
        """Generate complete HTML report content."""
        overall = evaluation_report["overall_metrics"]
        per_plate = evaluation_report.get("per_plate_type", {})

        html = self._build_header()
        html += self._build_styles()
        html += self._build_body_start()
        html += self._build_overall_metrics(overall)
        html += self._build_plate_type_summary(per_plate)
        html += self._build_filters(validation_results)
        html += self._build_results_grid(validation_results)
        html += self._build_scripts()
        html += self._build_footer()

        return html

    def _build_header(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ground Truth Evaluation Report</title>
"""

    def _build_styles(self) -> str:
        return """    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px 8px 0 0;
        }
        
        h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .metric-label {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #1f2937;
        }
        
        .metric-bar {
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .metric-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
        
        .filters {
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .filter-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .filter-group label {
            font-size: 14px;
            color: #6b7280;
            font-weight: 500;
        }
        
        .filter-group select, .filter-group input {
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .results-grid {
            padding: 30px;
        }
        
        .result-item {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }
        
        .result-item:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .result-header {
            background: #f9fafb;
            padding: 15px 20px;
            border-bottom: 1px solid #e5e7eb;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .result-title {
            font-weight: 600;
            color: #1f2937;
            font-size: 16px;
        }
        
        .result-subtitle {
            color: #6b7280;
            font-size: 14px;
            margin-top: 2px;
        }
        
        .result-scores {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .score-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
        }
        
        .score-high { background: #d1fae5; color: #065f46; }
        .score-medium { background: #fef3c7; color: #92400e; }
        .score-low { background: #fee2e2; color: #991b1b; }
        
        .result-body {
            padding: 20px;
            display: none;
        }
        
        .result-body.expanded {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .image-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .image-container {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
            background: #f9fafb;
        }
        
        .image-label {
            padding: 8px 12px;
            background: #f3f4f6;
            font-size: 13px;
            font-weight: 600;
            color: #4b5563;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .image-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .details-section {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .detail-group {
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
        }
        
        .detail-label {
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .ingredient-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .ingredient-tag {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 500;
        }
        
        .ingredient-correct { background: #d1fae5; color: #065f46; }
        .ingredient-expected { background: #dbeafe; color: #1e40af; }
        .ingredient-detected { background: #e0e7ff; color: #3730a6; }
        .ingredient-missed { background: #fee2e2; color: #991b1b; }
        .ingredient-extra { background: #fed7aa; color: #9a3412; }
        
        .metrics-detail {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        
        .metric-small {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 6px;
        }
        
        .metric-small-label {
            font-size: 12px;
            color: #6b7280;
        }
        
        .metric-small-value {
            font-size: 20px;
            font-weight: bold;
            color: #1f2937;
            margin-top: 4px;
        }
        
        .plate-type-summary {
            padding: 20px 30px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .plate-type-summary h2 {
            font-size: 20px;
            margin-bottom: 15px;
            color: #1f2937;
        }
        
        .plate-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .plate-card {
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        }
        
        .plate-name {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 8px;
        }
        
        .plate-stats {
            font-size: 13px;
            color: #6b7280;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .toggle-icon {
            transition: transform 0.2s;
        }
        
        .toggle-icon.expanded {
            transform: rotate(180deg);
        }
        
        @media (max-width: 768px) {
            .result-body.expanded {
                grid-template-columns: 1fr;
            }
            
            .metrics-summary {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
"""

    def _build_body_start(self) -> str:
        return """<body>
    <div class="container">
        <header>
            <h1>Ground Truth Evaluation Report</h1>
            <p>Food Analyzer - Detection & Classification Performance</p>
        </header>
"""

    def _build_overall_metrics(self, overall: Dict) -> str:
        return f"""        
        <div class="metrics-summary">
            <div class="metric-card">
                <div class="metric-label">Average Precision</div>
                <div class="metric-value">{overall['average_precision']:.3f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {overall['average_precision']*100}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Recall</div>
                <div class="metric-value">{overall['average_recall']:.3f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {overall['average_recall']*100}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average F1 Score</div>
                <div class="metric-value">{overall['average_f1']:.3f}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {overall['average_f1']*100}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Images</div>
                <div class="metric-value">{overall['total_images']}</div>
            </div>
        </div>
"""

    def _build_plate_type_summary(self, per_plate: Dict) -> str:
        if not per_plate:
            return ""

        html = """
        <div class="plate-type-summary">
            <h2>Performance by Plate Type</h2>
            <div class="plate-grid">
"""
        for plate_type, stats in sorted(per_plate.items()):
            html += f"""
                <div class="plate-card">
                    <div class="plate-name">{plate_type}</div>
                    <div class="plate-stats">
                        <div>Precision: {stats['precision']:.3f}</div>
                        <div>Recall: {stats['recall']:.3f}</div>
                        <div>F1: {stats['f1']:.3f}</div>
                        <div>Images: {stats['num_images']}</div>
                    </div>
                </div>
"""
        html += """
            </div>
        </div>
"""
        return html

    def _build_filters(self, validation_results: List[Dict]) -> str:
        html = """
        <div class="filters">
            <div class="filter-group">
                <label>Filter by Plate Type:</label>
                <select id="plateFilter" onchange="filterResults()">
                    <option value="">All</option>
"""

        plate_types = sorted(
            set(r["plate_type"] for r in validation_results if r["plate_type"] != "unknown")
        )
        for plate_type in plate_types:
            html += f'                    <option value="{plate_type}">{plate_type}</option>\n'

        html += """
                </select>
            </div>
            <div class="filter-group">
                <label>Min F1 Score:</label>
                <input type="number" id="f1Filter" min="0" max="1" step="0.1" value="0" onchange="filterResults()">
            </div>
            <div class="filter-group">
                <label>Search Ingredient:</label>
                <input type="text" id="ingredientFilter" placeholder="e.g., rice" oninput="filterResults()">
            </div>
        </div>
"""
        return html

    def _build_results_grid(self, validation_results: List[Dict]) -> str:
        html = """        
        <div class="results-grid" id="resultsGrid">
"""

        for result in validation_results:
            html += self._build_result_item(result)

        html += """
        </div>
"""
        return html

    def _build_result_item(self, result: Dict) -> str:
        image_name = result.get("image_name", "unknown")
        plate_type = result["plate_type"]
        f1 = result["f1"]
        precision = result["precision"]
        recall = result["recall"]

        score_class = "score-high" if f1 >= 0.7 else "score-medium" if f1 >= 0.4 else "score-low"

        expected = result.get("expected", [])
        detected = result.get("detected", [])
        missed = set(expected) - set(detected)
        extra = set(detected) - set(expected)
        correct = set(expected) & set(detected)

        image_stem = Path(image_name).stem
        original_path = f"../data/{image_name}"
        overlay_path = f"overlays/{image_stem}_overlay.png"

        original_img_data = self._get_image_data_url(original_path)
        overlay_img_data = self._get_image_data_url(self.results_dir / overlay_path)

        html = f"""
            <div class="result-item" data-plate="{plate_type}" data-f1="{f1}" data-ingredients="{','.join(expected + detected).lower()}">
                <div class="result-header" onclick="toggleResult(this)">
                    <div>
                        <div class="result-title">{image_name}</div>
                        <div class="result-subtitle">Plate Type: {plate_type} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}</div>
                    </div>
                    <div class="result-scores">
                        <span class="score-badge {score_class}">F1: {f1:.3f}</span>
                        <span class="toggle-icon">▼</span>
                    </div>
                </div>
                <div class="result-body">
                    <div class="image-section">
"""

        if original_img_data:
            html += f"""
                        <div class="image-container">
                            <div class="image-label">Original Image</div>
                            <img src="{original_img_data}" alt="Original">
                        </div>
"""

        if overlay_img_data:
            html += f"""
                        <div class="image-container">
                            <div class="image-label">Detections Overlay</div>
                            <img src="{overlay_img_data}" alt="Overlay">
                        </div>
"""

        html += f"""
                    </div>
                    <div class="details-section">
                        <div class="detail-group">
                            <div class="detail-label">Metrics</div>
                            <div class="metrics-detail">
                                <div class="metric-small">
                                    <div class="metric-small-label">Precision</div>
                                    <div class="metric-small-value">{precision:.3f}</div>
                                </div>
                                <div class="metric-small">
                                    <div class="metric-small-label">Recall</div>
                                    <div class="metric-small-value">{recall:.3f}</div>
                                </div>
                                <div class="metric-small">
                                    <div class="metric-small-label">F1 Score</div>
                                    <div class="metric-small-value">{f1:.3f}</div>
                                </div>
                            </div>
                        </div>
"""

        if correct:
            html += """
                        <div class="detail-group">
                            <div class="detail-label">Correctly Detected</div>
                            <div class="ingredient-list">
"""
            for ing in sorted(correct):
                html += f'                                <span class="ingredient-tag ingredient-correct">{ing}</span>\n'
            html += """
                            </div>
                        </div>
"""

        if missed:
            html += """
                        <div class="detail-group">
                            <div class="detail-label">Missed (False Negatives)</div>
                            <div class="ingredient-list">
"""
            for ing in sorted(missed):
                html += f'                                <span class="ingredient-tag ingredient-missed">{ing}</span>\n'
            html += """
                            </div>
                        </div>
"""

        if extra:
            html += """
                        <div class="detail-group">
                            <div class="detail-label">Extra Detections (False Positives)</div>
                            <div class="ingredient-list">
"""
            for ing in sorted(extra):
                html += f'                                <span class="ingredient-tag ingredient-extra">{ing}</span>\n'
            html += """
                            </div>
                        </div>
"""

        html += """
                    </div>
                </div>
            </div>
"""
        return html

    def _build_scripts(self) -> str:
        return """    
    <script>
        function toggleResult(header) {
            const body = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            body.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }
        
        function filterResults() {
            const plateFilter = document.getElementById('plateFilter').value.toLowerCase();
            const f1Filter = parseFloat(document.getElementById('f1Filter').value);
            const ingredientFilter = document.getElementById('ingredientFilter').value.toLowerCase();
            
            const items = document.querySelectorAll('.result-item');
            items.forEach(item => {
                const plate = item.getAttribute('data-plate').toLowerCase();
                const f1 = parseFloat(item.getAttribute('data-f1'));
                const ingredients = item.getAttribute('data-ingredients').toLowerCase();
                
                const plateMatch = !plateFilter || plate === plateFilter;
                const f1Match = f1 >= f1Filter;
                const ingredientMatch = !ingredientFilter || ingredients.includes(ingredientFilter);
                
                if (plateMatch && f1Match && ingredientMatch) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        }
    </script>
"""

    def _build_footer(self) -> str:
        return """</body>
</html>
"""

    def _get_image_data_url(self, image_path: Path | str) -> str:
        """Convert image to base64 data URL for embedding."""
        try:
            path = Path(image_path)
            if not path.exists():
                return ""

            with open(path, "rb") as f:
                image_data = f.read()

            encoded = base64.b64encode(image_data).decode("utf-8")
            ext = path.suffix.lower()
            mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

            return f"data:{mime_type};base64,{encoded}"
        except Exception:
            return ""
