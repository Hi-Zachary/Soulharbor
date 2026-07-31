# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/infer/chat_qwen3_nothink.py
# 原先用途: 命令行推理调试：Qwen3 + LoRA，关闭 thinking，用于训后抽查。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).lstrip()


def _load_system(system_path: str) -> str:
    text = Path(system_path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty system prompt: {system_path}")
    return text


def _maybe_load_adapter(model, adapter_name_or_path: Optional[str]):
    if not adapter_name_or_path:
        return model
    try:
        from peft import PeftModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing `peft`. Please `pip install peft`.") from e
    return PeftModel.from_pretrained(model, adapter_name_or_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Qwen3 chat with <think> stripped from outputs.")
    parser.add_argument("--model", type=str, required=True, help="Base model path, e.g. models/Qwen3-14B")
    parser.add_argument("--adapter", type=str, default="", help="LoRA adapter path (optional).")
    parser.add_argument("--system", type=str, default="prompts/system_soulharbor_zh.v1.txt")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit (bitsandbytes).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda:0", "cuda:1"],
        help="Force model to a single GPU (recommended for stable speed).",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation hint for transformers.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    system_text = _load_system(args.system)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quant_kwargs = {}
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing BitsAndBytesConfig. Please upgrade `transformers`.") from e

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **quant_kwargs,
    )

    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": int(args.device.split(":")[1])}

    # Best-effort: some model impls may not accept `attn_implementation`.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation=args.attn, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = _maybe_load_adapter(model, args.adapter or None)
    model.eval()

    history: List[Dict[str, str]] = [{"role": "system", "content": system_text}]

    print("Enter text. Ctrl-D / empty line to exit.")
    while True:
        try:
            user_text = input("\nUser> ").strip()
        except EOFError:
            break
        if not user_text:
            break

        history.append({"role": "user", "content": user_text})

        # Qwen3's built-in tokenizer chat_template may *insert* an empty <think>...</think>
        # when enable_thinking=False. We don't pass enable_thinking=False here.
        input_ids = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            bad_words_ids = []
            for bad in ("<think>", "</think>"):
                ids = tokenizer.encode(bad, add_special_tokens=False)
                if ids:
                    bad_words_ids.append(ids)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids or None,
            )

        gen_ids = output_ids[0][input_ids.shape[-1] :]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer = _strip_think(raw)

        print(f"\nAssistant> {answer}")
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
