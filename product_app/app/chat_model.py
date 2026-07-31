from __future__ import annotations

import re
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import TextIteratorStreamer

logger = logging.getLogger(__name__)


class CancelCriteria(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores=None, **kwargs) -> bool:
        return self.event.is_set()


_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

CHAT_ADAPTER = "chat"
TASK_ADAPTER = "task"


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).lstrip()


def _read_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    return text


def _pick_latest_checkpoint_dir(path: str) -> str:
    """Accept a checkpoint dir or a parent dir containing checkpoint-* subdirs."""
    p = Path(path)
    if not p.exists():
        return path
    if (p / "adapter_config.json").exists():
        return str(p)
    candidates: List[tuple[int, Path]] = []
    for sub in p.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", sub.name)
        if m and sub.is_dir():
            candidates.append((int(m.group(1)), sub))
    if not candidates:
        return str(p)
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


@dataclass
class DualLoRAConfig:
    model_path: str = "models/Qwen3-14B"
    chat_adapter_path: str = ""
    chat_consult_scale: float = 0.7
    chat_casual_scale: float = 0.3
    task_adapter_path: str = ""
    task_adapter_scale: float = 1.0
    summary_scale: float = 0.05
    system_path: str = "prompts/system_soulharbor_zh.txt"
    load_in_4bit: bool = True
    device: str = "auto"
    attn: str = "sdpa"
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 42


class DualLoRAQwenEngine:
    """
    Single 4-bit base + chat/task LoRA adapters with thread-safe switching.

    - consult route: DPO LoRA at chat_consult_scale (default 0.7)
    - casual route:  DPO LoRA at chat_casual_scale (default 0.3, light counseling tone)
    - background tasks: extraction_sft LoRA at task_adapter_scale (default 1.0)
    - memory rerank: chat (DPO) LoRA at chat_casual_scale (default 0.3), low temperature

    GPU access is serialized via a lock so chat replies are not corrupted by
    concurrent extraction/summary/rerank calls; post-turn work still runs in
    background threads and overlaps with user reading the reply.
    """

    def __init__(self, cfg: DualLoRAConfig) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        torch.manual_seed(cfg.seed)

        self.system_text = _read_text(cfg.system_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

        quant_kwargs: Dict[str, Any] = {}
        if cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                quant_kwargs = {}

        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **quant_kwargs,
        )
        if cfg.device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": int(cfg.device.split(":")[1])}

        try:
            base = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, attn_implementation=cfg.attn, **model_kwargs
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)

        chat_path = _pick_latest_checkpoint_dir(cfg.chat_adapter_path) if cfg.chat_adapter_path else ""
        task_path = _pick_latest_checkpoint_dir(cfg.task_adapter_path) if cfg.task_adapter_path else ""
        if not task_path:
            task_path = chat_path

        self._has_lora = bool(chat_path)
        self._task_is_separate = bool(task_path and task_path != chat_path)

        if self._has_lora:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, chat_path, adapter_name=CHAT_ADAPTER)
            if self._task_is_separate:
                self.model.load_adapter(task_path, adapter_name=TASK_ADAPTER)
        else:
            self.model = base

        self.model.eval()

    def _set_adapter_scale(self, adapter_name: str, scale: float) -> None:
        if not self._has_lora:
            return
        from peft.tuners.lora.layer import LoraLayer

        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.set_scale(adapter_name, scale)

    def _activate_chat(self, *, is_consult: int) -> None:
        if not self._has_lora:
            return
        self.model.set_adapter(CHAT_ADAPTER)
        scale = self.cfg.chat_consult_scale if is_consult else self.cfg.chat_casual_scale
        self._set_adapter_scale(CHAT_ADAPTER, scale)

    def _activate_task(self) -> None:
        if not self._has_lora:
            return
        adapter = TASK_ADAPTER if self._task_is_separate else CHAT_ADAPTER
        self.model.set_adapter(adapter)
        self._set_adapter_scale(adapter, self.cfg.task_adapter_scale)

    @torch.no_grad()
    def _build_history(self, messages: List[Dict[str, Any]], *, system_text: str = "") -> List[Dict[str, Any]]:
        base_system = system_text if system_text else self.system_text
        extra_system_parts: List[str] = []
        history: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "").strip()
            content = m.get("content") or ""
            if role == "system":
                if content.strip():
                    extra_system_parts.append(content.strip())
                continue
            if role not in ("user", "assistant", "tool"):
                continue
            entry: Dict[str, Any] = {"role": role, "content": content}
            if role == "tool" and m.get("tool_call_id"):
                entry["tool_call_id"] = m["tool_call_id"]
            history.append(entry)

        system_parts = [base_system.strip()] if base_system.strip() else []
        system_parts.extend(extra_system_parts)
        combined_system = "\n\n".join(system_parts)
        return [{"role": "system", "content": combined_system}, *history]

    def _bad_words_ids(self) -> List[List[int]] | None:
        bad_words_ids = []
        for bad in ("<think>", "</think>"):
            ids = self.tokenizer.encode(bad, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)
        return bad_words_ids or None

    def _prepare_model_inputs(self, messages: List[Dict[str, Any]], *, system_text: str = ""):
        history = self._build_history(messages, system_text=system_text)
        try:
            encoded = self.tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        if not isinstance(encoded, dict):
            encoded = dict(encoded)

        model_inputs = {k: v.to(self.model.device) for k, v in encoded.items()}
        return model_inputs

    @torch.no_grad()
    def _generate_once(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        system_text: str = "",
    ) -> str:
        model_inputs = self._prepare_model_inputs(messages, system_text=system_text)
        input_ids = model_inputs["input_ids"]
        mnt = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        temp = self.cfg.temperature if temperature is None else temperature

        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=mnt,
            do_sample=temp > 0,
            temperature=temp,
            top_p=self.cfg.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            bad_words_ids=self._bad_words_ids(),
            repetition_penalty=1.08,
        )

        gen_ids = output_ids[0][input_ids.shape[-1] :]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return _strip_think(raw)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        is_consult: int = 1,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        system_text: str = "",
    ) -> str:
        with self._lock:
            self._activate_chat(is_consult=is_consult)
            return self._generate_once(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_text=system_text,
            )

    def generate_chat_stream(self, messages: List[Dict[str, str]], *, is_consult: int = 1) -> Iterable[str]:
        with self._lock:
            self._activate_chat(is_consult=is_consult)
            model_inputs = self._prepare_model_inputs(messages)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            cancel_event = threading.Event()

            def _run() -> None:
                self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.temperature > 0,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=self._bad_words_ids(),
                    repetition_penalty=1.08,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([CancelCriteria(cancel_event)]),
                )

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            yield from self._iter_stream(streamer, worker, cancel_event)

    def _iter_stream(
        self,
        streamer: TextIteratorStreamer,
        worker: threading.Thread,
        cancel_event: threading.Event,
    ) -> Iterator[str]:
        try:
            for piece in streamer:
                if not piece:
                    continue
                if "◂think" in piece or "◂/think" in piece:
                    piece = piece.replace("◂think", "").replace("◂/think", "")
                yield piece
        finally:
            cancel_event.set()
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning("stream worker did not stop after cancel, detaching")

    def generate_task(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 384,
        temperature: float = 0.1,
        system_text: str = "",
    ) -> str:
        with self._lock:
            self._activate_task()
            return self._generate_once(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_text=system_text,
            )

    def generate_rerank(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        system_text: str = "",
    ) -> str:
        """Memory rerank: chat adapter at casual scale, not extraction_sft."""
        with self._lock:
            self._activate_chat(is_consult=0)
            return self._generate_once(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_text=system_text,
            )

    def generate_summary(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        system_text: str = "",
    ) -> str:
        """Summary generation: chat adapter at very low scale (close to base)."""
        with self._lock:
            if self._has_lora:
                self.model.set_adapter(CHAT_ADAPTER)
                self._set_adapter_scale(CHAT_ADAPTER, self.cfg.summary_scale)
            return self._generate_once(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_text=system_text,
            )

    def generate_base(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        system_text: str = "",
    ) -> str:
        """Pure base-model generation with all LoRA adapters disabled."""
        with self._lock:
            if self._has_lora and hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    return self._generate_once(
                        messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        system_text=system_text,
                    )
            return self._generate_once(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_text=system_text,
            )


# Backward-compatible alias for scripts that still import LocalQwen3Chat.
@dataclass
class ChatConfig:
    model_path: str = "models/Qwen3-14B"
    adapter_path: str = ""
    adapter_scale: float = 1.0
    system_path: str = "prompts/system_soulharbor_zh.txt"
    load_in_4bit: bool = True
    device: str = "auto"
    attn: str = "sdpa"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 42


class LocalQwen3Chat:
    """Legacy single-adapter wrapper; prefer DualLoRAQwenEngine in production."""

    def __init__(self, cfg: ChatConfig) -> None:
        self._engine = DualLoRAQwenEngine(
            DualLoRAConfig(
                model_path=cfg.model_path,
                chat_adapter_path=cfg.adapter_path,
                chat_consult_scale=cfg.adapter_scale,
                chat_casual_scale=cfg.adapter_scale,
                system_path=cfg.system_path,
                load_in_4bit=cfg.load_in_4bit,
                device=cfg.device,
                attn=cfg.attn,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed,
            )
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        system_text: str = "",
    ) -> str:
        return self._engine.generate_chat(
            messages,
            is_consult=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_text=system_text,
        )

    def generate_stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        return self._engine.generate_chat_stream(messages, is_consult=1)
