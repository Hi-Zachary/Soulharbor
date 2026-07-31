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
    llm_extraction_adapter: str = _abs(
        os.environ.get("SOULHARBOR_LLM_EXTRACTION_ADAPTER", "")
    )
    llm_extraction_adapter_scale: float = float(
        os.environ.get("SOULHARBOR_LLM_EXTRACTION_ADAPTER_SCALE", "1.0")
    )
    llm_summary_adapter_scale: float = float(
        os.environ.get("SOULHARBOR_LLM_SUMMARY_SCALE", "0.05")
    )
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
    # Default cuda; on small GPUs set SOULHARBOR_MEMORY_EMBED_DEVICE=cpu etc.
    memory_embed_device: str = os.environ.get("SOULHARBOR_MEMORY_EMBED_DEVICE", "cuda")
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
    # Total active memory node soft-cap (legacy profile/preference/situation budgets collapsed).
    memory_max_per_user: int = int(os.environ.get("SOULHARBOR_MEMORY_MAX_PER_USER", "100"))
    memory_max_profile: int = int(os.environ.get("SOULHARBOR_MEMORY_MAX_PROFILE", "10"))
    memory_max_preference: int = int(os.environ.get("SOULHARBOR_MEMORY_MAX_PREFERENCE", "10"))
    memory_max_situation: int = int(os.environ.get("SOULHARBOR_MEMORY_MAX_SITUATION", "80"))
    memory_summary_window_max_tokens: int = int(
        os.environ.get("SOULHARBOR_MEMORY_SUMMARY_WINDOW_MAX_TOKENS", "3000")
    )
    memory_relevant_limit: int = int(os.environ.get("SOULHARBOR_MEMORY_RELEVANT_LIMIT", "5"))
    memory_max_inject_chars: int = int(os.environ.get("SOULHARBOR_MEMORY_MAX_INJECT_CHARS", "3600"))
    memory_always_on_chars: int = int(os.environ.get("SOULHARBOR_MEMORY_ALWAYS_ON_CHARS", "2400"))
    memory_relevant_chars: int = int(os.environ.get("SOULHARBOR_MEMORY_RELEVANT_CHARS", "1200"))
    memory_score_threshold: float = float(os.environ.get("SOULHARBOR_MEMORY_SCORE_THRESHOLD", "0.35"))
    memory_use_llm_rerank: bool = os.environ.get("SOULHARBOR_MEMORY_USE_LLM_RERANK", "1") != "0"
    memory_dedup_threshold: float = float(os.environ.get("SOULHARBOR_MEMORY_DEDUP_THRESHOLD", "0.9"))
    memory_extraction_min_tokens: int = int(
        os.environ.get("SOULHARBOR_MEMORY_EXTRACTION_MIN_TOKENS", "800")
    )
    # Forgetting-curve decay (strength * exp(-age/tau)); higher layers decay slower.
    memory_decay_tau_l1: float = float(os.environ.get("SOULHARBOR_MEMORY_DECAY_TAU_L1", "14"))
    memory_decay_tau_l2: float = float(os.environ.get("SOULHARBOR_MEMORY_DECAY_TAU_L2", "45"))
    memory_decay_tau_l3: float = float(os.environ.get("SOULHARBOR_MEMORY_DECAY_TAU_L3", "120"))
    memory_close_threshold: float = float(os.environ.get("SOULHARBOR_MEMORY_CLOSE_THRESHOLD", "0.18"))
    memory_touch_boost_l1: float = float(os.environ.get("SOULHARBOR_MEMORY_TOUCH_BOOST_L1", "0.08"))
    memory_touch_boost_l2: float = float(os.environ.get("SOULHARBOR_MEMORY_TOUCH_BOOST_L2", "0.05"))
    memory_touch_boost_l3: float = float(os.environ.get("SOULHARBOR_MEMORY_TOUCH_BOOST_L3", "0.03"))
    # Dream consolidation gate (Claude Code style: hours + sessions).
    memory_dream_min_hours: float = float(os.environ.get("SOULHARBOR_MEMORY_DREAM_MIN_HOURS", "24"))
    memory_dream_min_sessions: int = int(os.environ.get("SOULHARBOR_MEMORY_DREAM_MIN_SESSIONS", "5"))
    # Legacy aliases kept for older callers.
    memory_decay_floor: float = float(os.environ.get("SOULHARBOR_MEMORY_DECAY_FLOOR", "0.3"))
    memory_decay_tau_days: float = float(os.environ.get("SOULHARBOR_MEMORY_DECAY_TAU_DAYS", "30"))
    memory_faiss_top_k: int = int(os.environ.get("SOULHARBOR_MEMORY_FAISS_TOP_K", "64"))
    # L2 cross-encoder rerank (between dense recall and LLM rerank).
    memory_reranker_base: str = _abs(
        os.environ.get(
            "SOULHARBOR_MEMORY_RERANKER_BASE",
            "models/encoders/bge-reranker-v2-m3",
        )
    )
    memory_use_cross_rerank: bool = (
        os.environ.get("SOULHARBOR_MEMORY_USE_CROSS_RERANK", "1") != "0"
    )
    memory_cross_rerank_limit: int = int(
        os.environ.get("SOULHARBOR_MEMORY_CROSS_RERANK_LIMIT", "10")
    )
    memory_reranker_device: str = os.environ.get("SOULHARBOR_MEMORY_RERANKER_DEVICE", "cpu")


settings = Settings()
