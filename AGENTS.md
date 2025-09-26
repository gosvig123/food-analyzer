# Repository Guidelines

## Project Structure & Module Organization
`food_analyzer/` houses the package modules: `core/pipeline.py` orchestrates detection → classification → nutrition lookups via types in `core/types.py`; `detection/` wraps Torchvision detectors and optional Segment Anything refinement; `classification/` manages CLIP-based classifiers, ingredient APIs, and dynamic label logic; `data/` centralizes label maps; `utils/` contains config and label helpers; `optimization/parameter_optimizer.py` tunes detector/classifier settings. Supporting directories include `data/` for input images, `models/` for checkpoints, `results/` for JSON, CSV, and overlay artifacts, plus root-level `config.json`, `ground_truth.json`, and `ingredient_config.json`. Use `main.py` as the CLI entry point.

## Build, Test, and Development Commands
Create a virtual environment with `python -m venv .venv && source .venv/bin/activate`, then install dependencies surfaced in the modules (`pip install torch torchvision pillow numpy opencv-python segment-anything`). Run inference locally via `python main.py data/sample_images --results-dir results/dev_run`. Compare historical runs with `python main.py data/sample_images --compare-dirs results/runA results/runB --compare-out results/compare.csv`. Kick off parameter search when evaluation data exists using `python main.py data/sample_images --optimize --evaluation-file results/latest_evaluation.json`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and keep type hints on public functions. Use concise docstrings mirroring existing modules. Stick to snake_case for files, PascalCase for classes, and UPPER_SNAKE_CASE for constants. Favor dataclasses for structured payloads, keep GPU-dependent logic behind capability checks, and route orchestration through `FoodInferencePipeline` instead of the CLI.

## Testing Guidelines
No automated test suite exists yet. Validate changes by running the CLI on representative image sets and reviewing JSON payloads, overlays, and crops written to `results/`. When `ground_truth.json` is populated, execute `python main.py data/plates --results-dir results/run --config config.json` to emit precision/recall reports at `results/run/ground_truth_evaluation.json`. Document manual test coverage or notable result diffs in your PR description.

## Commit & Pull Request Guidelines
Commit history favors capitalized, imperative subject lines (`Remove nutrition estimation`, `Improve detection accuracy`). Keep the subject under ~72 characters and add body bullets if rationale is complex. PRs should outline the change, reference any configs touched, attach validation outputs (paths in `results/` or console snippets), and link related issues. Include overlay or crop screenshots when model behavior changes visibly.
