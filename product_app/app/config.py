from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


_ROOT = Path(__file__).resolve().parents[2]  # SoulHarbor repo root


def _abs(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((_ROOT / pp).resolve())


@dataclass(frozen=True)
class Settings:
    llm_base: str = _abs(os.environ.get("SOULHARBOR_LLM_BASE", "models/Qwen3-14B"))
    llm_adapter: str = _abs(
        os.environ.get(
            "SOULHARBOR_LLM_ADAPTER",
            "saves/qwen14b/lora/dpo_synth_20260425_060332",
        )
    )
    llm_adapter_scale: float = float(os.environ.get("SOULHARBOR_LLM_ADAPTER_SCALE", "0.7"))
    llm_casual_adapter_scale: float = float(
        os.environ.get("SOULHARBOR_LLM_CASUAL_ADAPTER_SCALE", "0.3")
    )
    llm_max_new_tokens: int = int(os.environ.get("SOULHARBOR_LLM_MAX_NEW_TOKENS", "1024"))
    llm_system: str = _abs(os.environ.get("SOULHARBOR_LLM_SYSTEM", "prompts/system_soulharbor_zh.txt"))
    llm_load_4bit: bool = os.environ.get("SOULHARBOR_LLM_4BIT", "1") != "0"
    llm_device: str = os.environ.get("SOULHARBOR_LLM_DEVICE", "auto")  # auto/cuda:0/cuda:1

    classifier_run: str = _abs(
        os.environ.get("SOULHARBOR_CLASSIFIER_RUN", "outputs/classifiers")
    )
    encoder_base: str = _abs(os.environ.get("SOULHARBOR_ENCODER_BASE", "models/encoders/chinese-macbert-base"))
    memory_encoder_base: str = _abs(
        os.environ.get("SOULHARBOR_MEMORY_ENCODER_BASE", "models/encoders/bge-m3")
    )
    memory_reranker_base: str = _abs(
        os.environ.get(
            "SOULHARBOR_MEMORY_RERANKER_BASE",
            "models/encoders/bge-reranker-v2-m3",
        )
    )
    # Default cuda; on small GPUs set SOULHARBOR_MEMORY_EMBED_DEVICE=cpu etc.
    memory_embed_device: str = os.environ.get("SOULHARBOR_MEMORY_EMBED_DEVICE", "cuda")
    memory_rerank_device: str = os.environ.get(
        "SOULHARBOR_MEMORY_RERANK_DEVICE",
        os.environ.get("SOULHARBOR_MEMORY_EMBED_DEVICE", "cuda"),
    )
    classifier_device: str = os.environ.get("SOULHARBOR_CLASSIFIER_DEVICE", "cuda")
    memory_embed_max_length: int = int(os.environ.get("SOULHARBOR_MEMORY_EMBED_MAXLEN", "512"))
    max_length: int = int(os.environ.get("SOULHARBOR_CLS_MAXLEN", "512"))
    db_path: str = _abs(os.environ.get("SOULHARBOR_DB_PATH", "product_app/data/soulharbor.db"))
    admin_password: str = os.environ.get("SOULHARBOR_ADMIN_PASSWORD", "soulharbor_admin")
    post_turn_workers: int = int(os.environ.get("SOULHARBOR_POST_TURN_WORKERS", "8"))

    memory_enabled: bool = os.environ.get("SOULHARBOR_MEMORY_ENABLED", "1") != "0"
    # Keep recent window and summary kickoff aligned to avoid a blind gap
    # (messages fallen out of the window before any session summary exists).
    memory_recent_turns: int = int(os.environ.get("SOULHARBOR_MEMORY_RECENT_TURNS", "8"))
    memory_summary_window_max_tokens: int = int(
        os.environ.get("SOULHARBOR_MEMORY_SUMMARY_WINDOW_MAX_TOKENS", "3000")
    )


settings = Settings()
