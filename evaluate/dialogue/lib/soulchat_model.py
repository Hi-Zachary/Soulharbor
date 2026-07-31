from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_pad(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_model_and_tokenizer(
    *,
    base: str,
    adapter: str,
    device: str,
    attn: str,
    load_in_4bit: bool,
) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, padding_side="left")
    ensure_pad(tokenizer)

    quant_kwargs: Dict[str, Any] = {}
    if load_in_4bit:
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
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": int(device.split(":")[1])}

    try:
        model = AutoModelForCausalLM.from_pretrained(base, attn_implementation=attn, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer

