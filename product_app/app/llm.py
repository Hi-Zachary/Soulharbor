from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from product_app.app.chat_model import DualLoRAConfig, DualLoRAQwenEngine
from product_app.app.generation_log import GenerationLog


@dataclass(frozen=True)
class LLMConfig:
    base: str
    adapter: str
    adapter_scale: float
    casual_adapter_scale: float
    system: str
    load_4bit: bool
    device: str
    extraction_adapter: str = ""
    extraction_adapter_scale: float = 1.0
    summary_adapter_scale: float = 0.05
    max_new_tokens: int = 1024


class SoulHarborLLM:
    """
    Route-aware chat generation + dedicated extraction LoRA on one shared base.

    consult → DPO LoRA at adapter_scale (default 0.7)
    chat    → DPO LoRA at casual_adapter_scale (default 0.3, light counseling tone)
    tasks   → extraction_sft LoRA (extract / summary)
    rerank  → chat LoRA at casual_adapter_scale (default 0.3), structured id output
    base    → all LoRA disabled
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._engine = DualLoRAQwenEngine(
            DualLoRAConfig(
                model_path=cfg.base,
                chat_adapter_path=cfg.adapter,
                chat_consult_scale=cfg.adapter_scale,
                chat_casual_scale=cfg.casual_adapter_scale,
                task_adapter_path=cfg.extraction_adapter or cfg.adapter,
                task_adapter_scale=cfg.extraction_adapter_scale,
                summary_scale=cfg.summary_adapter_scale,
                system_path=cfg.system,
                load_in_4bit=cfg.load_4bit,
                device=cfg.device,
                max_new_tokens=cfg.max_new_tokens,
            )
        )

    @staticmethod
    def _log(
        *,
        route: str,
        messages: List[Dict[str, str]],
        output: str,
        system_text: str = "",
        max_new_tokens: int | None = None,
        **extra: object,
    ) -> None:
        GenerationLog.record(
            route=route,
            messages=list(messages or []),
            output=output or "",
            system_text=system_text or "",
            max_new_tokens=max_new_tokens,
            **extra,
        )

    def generate(self, messages: List[Dict[str, str]], *, is_consult: int = 1) -> str:
        out = self._engine.generate_chat(messages, is_consult=is_consult)
        self._log(route="chat", messages=messages, output=out, is_consult=is_consult)
        return out

    def generate_task(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 256,
        system_text: str = "",
    ) -> str:
        out = self._engine.generate_task(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            system_text=system_text,
        )
        self._log(
            route="task",
            messages=messages,
            output=out,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
        )
        return out

    def generate_rerank(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 128,
        system_text: str = "",
    ) -> str:
        out = self._engine.generate_rerank(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            system_text=system_text,
        )
        self._log(
            route="rerank",
            messages=messages,
            output=out,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
        )
        return out

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 256,
        system_text: str = "",
    ) -> str:
        out = self._engine.generate_rerank(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            system_text=system_text,
        )
        self._log(
            route="structured",
            messages=messages,
            output=out,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
        )
        return out

    def generate_base(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 512,
        system_text: str = "",
    ) -> str:
        """Memory Gate / consolidation: base model only, no LoRA."""
        out = self._engine.generate_base(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            system_text=system_text,
        )
        self._log(
            route="base",
            messages=messages,
            output=out,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
        )
        return out

    def generate_summary(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 512,
        system_text: str = "",
    ) -> str:
        out = self._engine.generate_summary(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            system_text=system_text,
        )
        self._log(
            route="summary",
            messages=messages,
            output=out,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
        )
        return out

    def generate_stream(self, messages: List[Dict[str, str]], *, is_consult: int = 1) -> Iterable[str]:
        return self._engine.generate_chat_stream(messages, is_consult=is_consult)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._engine.tokenizer.encode(text, add_special_tokens=False))
