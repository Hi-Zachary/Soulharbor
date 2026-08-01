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
    max_new_tokens: int = 1024


class SoulHarborLLM:
    """
    Route-aware chat generation on one shared 4-bit base + chat LoRA.

    consult    → DPO LoRA at adapter_scale (default 0.7)
    casual     → DPO LoRA at casual_adapter_scale (default 0.3)
    structured → chat LoRA at casual scale, low temperature (query planner)
    summary    → base model, adapters disabled
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._engine = DualLoRAQwenEngine(
            DualLoRAConfig(
                model_path=cfg.base,
                chat_adapter_path=cfg.adapter,
                chat_consult_scale=cfg.adapter_scale,
                chat_casual_scale=cfg.casual_adapter_scale,
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

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 256,
        system_text: str = "",
    ) -> str:
        out = self._engine.generate_structured(
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
