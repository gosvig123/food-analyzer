"""Whole-image dish classifier with configurable, file-driven label sets.

No hard-coded dish labels. Labels must be provided via config or file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import torch
from PIL import Image


@dataclass
class DishClassifierConfig:
    backend: str = "clip_zeroshot"
    labels_path: str | None = None  # JSON or TXT (one label per line)
    topk: int = 5
    min_conf: float = 0.05
    device: str | None = None
    model_name: str | None = None
    pretrained: str | None = None


def _load_labels_from_path(path: str) -> list[str]:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dish labels file not found: {p}")
    if p.suffix.lower() in {".json"}:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        raise ValueError("Dish labels JSON must be a list of strings")
    # Fallback: text lines
    labels: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            labels.append(line)
    if not labels:
        raise ValueError("No dish labels found in file")
    return labels


class DishClassifier:
    """Zero-shot dish classifier using CLIP via open-clip.

    - Requires dish labels supplied externally (file/config).
    - Returns a list of {label, confidence} dicts sorted desc by confidence.
    """

    def __init__(self, cfg: DishClassifierConfig, labels: list[str]):
        self.cfg = cfg
        self.labels = labels
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.backend != "clip_zeroshot":
            raise ValueError(f"Unsupported dish backend: {cfg.backend}")
        try:
            import open_clip  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"open-clip-torch not available: {exc}")

        model_name = cfg.model_name or "ViT-L-14-336"
        pretrained = cfg.pretrained or "openai"
        self.clip_model, _, preprocess = open_clip.create_model_and_transforms(  # type: ignore
            model_name, pretrained=pretrained
        )
        self.clip_model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)  # type: ignore

        if not self.labels:
            raise ValueError("DishClassifier requires non-empty labels list")

        # Build text features once
        templates = [
            "a photo of {}",
            "a dish of {}",
            "a meal of {}",
            "{} on a plate",
            "a serving of {}",
            "{} dish",
            "traditional {}",
            "{} food",
        ]
        with torch.inference_mode():
            feats = []
            for t in templates:
                prompts = [t.format(lbl) for lbl in self.labels]
                text = self.tokenizer(prompts)
                f = self.clip_model.encode_text(text.to(self.device))  # type: ignore
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f)
            text_features = sum(feats) / len(feats)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @classmethod
    def from_config(cls, cfg_dict: dict) -> "DishClassifier":
        cfg = DishClassifierConfig(
            backend=str(cfg_dict.get("backend", "clip_zeroshot")),
            labels_path=cfg_dict.get("labels_path"),
            topk=int(cfg_dict.get("topk", 5)),
            min_conf=float(cfg_dict.get("min_conf", 0.05)),
            device=cfg_dict.get("device"),
            model_name=cfg_dict.get("model_name"),
            pretrained=cfg_dict.get("pretrained"),
        )
        labels: list[str] = []
        if isinstance(cfg.labels_path, str):
            labels = _load_labels_from_path(cfg.labels_path)
        else:
            # No hard-coded fallback; require labels file.
            raise ValueError("DishClassifier requires labels_path in config; no hard-coded labels allowed")
        return cls(cfg, labels)

    def __call__(self, image: Image.Image) -> list[dict]:
        image = image.convert("RGB")
        with torch.inference_mode():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(tensor)  # type: ignore
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * (image_features @ self.text_features.T)  # type: ignore
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        # Top-k selection limited by available labels
        k = min(self.cfg.topk, len(self.labels))
        vals, idxs = torch.topk(probs, k)
        results: list[dict] = []
        for v, i in zip(vals, idxs):
            conf = float(v.item())
            if conf < self.cfg.min_conf:
                continue
            label = self.labels[int(i.item())]
            results.append({"label": label, "confidence": conf})
        return results
