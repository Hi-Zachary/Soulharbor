"""BGE-M3 embedder used by memory search."""
from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

import torch

from product_app.app.config import settings


def _model_path(name: str) -> str:
    path = Path(name)
    if path.is_dir() and (path / "config.json").exists():
        return str(path.resolve())
    return name


def _drop_local_datasets_shadow() -> None:
    """Avoid a repo-local `datasets/` package shadowing HuggingFace datasets."""
    shadowed: List[str] = []
    for entry in list(sys.path):
        root = entry or os.getcwd()
        try:
            init_file = Path(root).resolve() / "datasets" / "__init__.py"
        except Exception:
            continue
        if not init_file.exists():
            continue
        if "site-packages" in str(init_file):
            continue
        shadowed.append(entry)

    if not shadowed:
        return

    sys.path[:] = [p for p in sys.path if p not in shadowed]
    loaded = sys.modules.get("datasets")
    loaded_file = str(getattr(loaded, "__file__", "") or "")
    if loaded is not None and loaded_file and "site-packages" not in loaded_file:
        sys.modules.pop("datasets", None)
    importlib.invalidate_caches()


class MemoryEmbedder:
    _lock = threading.Lock()
    _shared: Optional["MemoryEmbedder"] = None

    def __init__(self) -> None:
        self.model_path = _model_path(settings.memory_encoder_base)
        pref = str(getattr(settings, "memory_embed_device", "cpu") or "cpu").lower()
        if pref in {"cuda", "gpu"} and torch.cuda.is_available():
            self._device = "cuda"
        elif pref.startswith("cuda:") and torch.cuda.is_available():
            self._device = pref
        else:
            self._device = "cpu"
        self._use_fp16 = self._device.startswith("cuda")
        self._model = None
        self._load_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "MemoryEmbedder":
        with cls._lock:
            if cls._shared is None:
                cls._shared = MemoryEmbedder()
            return cls._shared

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._shared = None

    @property
    def embed_dim(self) -> int:
        return 1024

    @property
    def model_id(self) -> str:
        return self.model_path

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            _drop_local_datasets_shadow()
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_path,
                use_fp16=self._use_fp16,
                devices=self._device,
            )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cleaned = [(t or "").strip() for t in texts]
        if not any(cleaned):
            return []
        self._ensure_loaded()
        encoded = self._model.encode(  # type: ignore[union-attr]
            cleaned,
            batch_size=min(8, max(1, len(cleaned))),
            max_length=settings.memory_embed_max_length,
        )
        dense = encoded["dense_vecs"]
        return [[float(x) for x in vec.tolist()] for vec in dense]

    def embed(self, text: str) -> List[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        vectors = self.embed_batch([cleaned])
        return vectors[0] if vectors else []


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = sum(a * a for a in left) ** 0.5
    norm_r = sum(b * b for b in right) ** 0.5
    if norm_l <= 0 or norm_r <= 0:
        return 0.0
    return float(dot / (norm_l * norm_r))
