from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _pick_checkpoint_dir(model_dir: Path) -> Path:
    """Prefer validation-best checkpoint from trainer_state; else latest step."""
    candidates: List[Tuple[int, Path]] = []
    for p in model_dir.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* found in {model_dir}")
    candidates.sort(key=lambda x: x[0])

    # Prefer best_model_checkpoint recorded by HF Trainer (load_best_model_at_end).
    by_step = {step: path for step, path in candidates}
    for _, ckpt in reversed(candidates):
        state_path = ckpt / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best_step = state.get("best_global_step")
        if isinstance(best_step, int) and best_step in by_step:
            return by_step[best_step]
        best_path = state.get("best_model_checkpoint")
        if isinstance(best_path, str) and best_path:
            name = Path(best_path).name
            m = re.match(r"checkpoint-(\d+)$", name)
            if m and int(m.group(1)) in by_step:
                return by_step[int(m.group(1))]

    return candidates[-1][1]


@dataclass(frozen=True)
class IntentResult:
    is_consult: int
    prob: float


class IntentClassifier:
    def __init__(self, *, run_dir: str, encoder_base: str, max_length: int = 512, device: Optional[str] = None) -> None:
        self.run_dir = Path(run_dir)
        self.encoder_base = encoder_base
        self.max_length = int(max_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = _pick_checkpoint_dir(self.run_dir / "intent")
        self.checkpoint_dir = ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> IntentResult:
        t = (text or "").strip()
        enc = self.tokenizer([t], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(torch.argmax(probs).item())
        # label "1" == is_consult=1 (see build_classifier_dataset_v2.py)
        return IntentResult(is_consult=pred, prob=float(probs[pred].item()))
