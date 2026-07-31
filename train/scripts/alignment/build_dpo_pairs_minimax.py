# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/build_dpo_pairs_minimax.py
# 原先用途: 用 MiniMax 裁判生成 chosen/rejected，写出 DPO JSONL。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl_append(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(str(obj.get("id") or ""))
            except Exception:
                continue
    done.discard("")
    return done


def _load_env_file(path: Path) -> None:
    """
    Load simple KEY=VALUE env file (no python-dotenv dependency).
    - ignores empty lines and # comments
    - does NOT override existing os.environ
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Support a single-line file that only contains an API key.
        # (Some users store just the key in .env without KEY=VALUE format.)
        if "=" not in line:
            if line.startswith("http://") or line.startswith("https://"):
                os.environ.setdefault("ANTHROPIC_BASE_URL", line)
            else:
                # Do not assume key prefix (not all providers use "sk-").
                os.environ.setdefault("ANTHROPIC_API_KEY", line)
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        os.environ.setdefault(k, v)


def _anthropic_messages_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("Empty base_url")
    # Accept either:
    # - root (..../anthropic) -> ..../anthropic/v1/messages
    # - v1 root (..../anthropic/v1) -> ..../anthropic/v1/messages
    # - full messages URL (..../anthropic/v1/messages) -> unchanged
    if b.endswith("/v1/messages"):
        return b
    if b.endswith("/v1"):
        return b + "/messages"
    return b + "/v1/messages"


def _call_anthropic_compat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    timeout_s: int,
    include_thinking: bool = False,
) -> str:
    url = _anthropic_messages_url(base_url)
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        # Some compatible gateways accept Bearer too; harmless if ignored.
        "authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system or "",
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    }
    if top_p is not None:
        payload["top_p"] = float(top_p)

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    parts: List[str] = []
    content = data.get("content")
    if isinstance(content, list):
        for blk in content:
            if not isinstance(blk, dict):
                continue
            t = str(blk.get("type") or "")
            if t == "text":
                parts.append(str(blk.get("text") or blk.get("content") or blk.get("value") or ""))
            elif include_thinking and t == "thinking":
                # MiniMax may return content only in thinking for constrained outputs.
                parts.append(str(blk.get("thinking") or blk.get("text") or blk.get("content") or ""))
            else:
                if blk.get("text"):
                    parts.append(str(blk.get("text") or ""))
                elif blk.get("content"):
                    parts.append(str(blk.get("content") or ""))
                elif include_thinking and blk.get("thinking"):
                    parts.append(str(blk.get("thinking") or ""))
    elif isinstance(data.get("choices"), list):
        # Some gateways respond with an OpenAI-like schema.
        for ch in data.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg.get("content") or ""))
            elif ch.get("text"):
                parts.append(str(ch.get("text") or ""))
    else:
        for k in ("output_text", "completion", "text", "answer", "result"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v)
                break
    return "\n".join([p for p in parts if p]).strip()


def _with_retry(fn, *, retries: int, sleep_s: float, overload_wait_s: float) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e)
            wait = sleep_s * (2**i)
            # MiniMax gateways sometimes return 529 for overload; wait longer.
            if "HTTP 529" in msg or "overloaded" in msg.lower():
                wait = max(wait, overload_wait_s)
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {last_err}")


def _format_dialog(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
    return "\n".join(lines).strip()


def _to_sharegpt_conversations(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    conv: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            conv.append({"from": "human", "value": content})
        elif role == "assistant":
            conv.append({"from": "gpt", "value": content})
    # Ensure prompt ends with human.
    while conv and conv[-1]["from"] != "human":
        conv.pop()
    return conv


_RE_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_winner(text: str) -> str:
    """
    Judge output can be:
    - strict JSON: {"winner":"A"} (preferred)
    - a bare letter: A / B (common in practice)
    - JSON wrapped in markdown fences
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty judge output.")

    # Normalize fullwidth letters.
    t = t.translate(str.maketrans({"Ａ": "A", "Ｂ": "B", "ａ": "A", "ｂ": "B"}))

    # Fast path: sometimes gateways return exactly "A" / "B" with a trailing newline.
    t_simple = re.sub(r"[`\s]+", "", t).strip().upper()
    if t_simple in ("A", "B"):
        return t_simple

    # Try JSON objects (some models output {"winner":"A"}).
    for m in _RE_JSON_OBJ.finditer(t):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict):
            w = str(obj.get("winner") or "").strip().upper()
            if w in ("A", "B"):
                return w

    # Fallback 1: last non-empty line is often just "A" / "B".
    lines = [ln.strip().upper() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = re.sub(r"^[^AB]*|[^AB]*$", "", lines[-1])
        if last in ("A", "B"):
            return last

    # Fallback 2: find an A/B token in the response.
    # IMPORTANT: some gateways echo the prompt (which contains "候选回复 A/B"),
    # so we intentionally take the *last* match instead of the earliest.
    t2 = t.replace("`", " ").replace("：", ":").strip().upper()
    # Common patterns like "选A" / "选择A" / "A更好".
    for pat, w in (
        (r"选择\s*A", "A"),
        (r"选择\s*B", "B"),
        (r"选\s*A", "A"),
        (r"选\s*B", "B"),
    ):
        ms = list(re.finditer(pat, t2))
        if ms:
            return w
    matches = list(re.finditer(r"(^|[^A-Z])([AB])([^A-Z]|$)", t2))
    if matches:
        return matches[-1].group(2)

    raise ValueError("Cannot parse winner from judge output.")


@dataclass(frozen=True)
class GenCfg:
    temperature: float
    top_p: Optional[float]


GEN_A = GenCfg(temperature=0.2, top_p=0.9)
GEN_B = GenCfg(temperature=0.8, top_p=0.95)


SYSTEM_ASSISTANT = (
    "你是校园心理健康助手。用温和、具体、自然的中文回应。"
    "安全边界：不提供任何自伤/自杀的方法、步骤、工具、剂量或可操作细节。"
)

SYSTEM_JUDGE = (
    "你是对话回复质量评审。你将看到一段对话上下文，以及两个候选回复 A/B。"
    "请根据“共情贴合度、自然程度、不过度追问、避免复读、对自伤自杀不提供方法细节”选择更好的一个。"
    "只输出一个大写字母 A 或 B（不要输出解释）。"
)

SYSTEM_JUDGE_STRICT = "你是评审。只输出一个字符：A 或 B。不要输出任何其它字符、标点、换行或解释。"


def _simple_score(text: str) -> float:
    """
    Fallback scorer used ONLY when the judge output is unparsable.
    Goal: prefer non-empty, non-repetitive, not-too-question-heavy answers.
    """
    t = (text or "").strip()
    if not t:
        return -1e9

    length = len(t)
    length_score = min(length, 400) / 400.0  # 0..1

    q = t.count("?") + t.count("？")
    q_pen = min(q, 6) * 0.06  # up to -0.36

    # 3-gram repetition ratio
    s = re.sub(r"\s+", "", t)
    if len(s) < 12:
        rep_pen = 0.0
    else:
        grams = [s[i : i + 3] for i in range(0, len(s) - 2)]
        total = len(grams)
        uniq = len(set(grams))
        rep_ratio = 1.0 - (uniq / max(total, 1))
        rep_pen = min(rep_ratio, 0.4) * 1.2  # up to -0.48

    return length_score - q_pen - rep_pen

DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build synthetic DPO pairs using MiniMax (Anthropic-compatible).")
    ap.add_argument("--input", default="archive/llm_build/dpo_prompts_pool.jsonl")
    ap.add_argument("--output", default="data/llm/dpo_synth_minimax.jsonl")
    ap.add_argument("--max-prompts", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--overload-wait", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--env-file",
        default="",
        help="Optional .env file to load MINIMAX/ANTHROPIC credentials (will not override existing env).",
    )
    ap.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"))
    ap.add_argument("--judge-model", default=os.environ.get("MINIMAX_JUDGE_MODEL", "MiniMax-M2.7"))
    ap.add_argument("--system-assistant", default=SYSTEM_ASSISTANT)
    ap.add_argument("--system-judge", default=SYSTEM_JUDGE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    if args.env_file:
        _load_env_file(Path(args.env_file))
    else:
        default_env = Path("secrets/minimax.env")
        if default_env.exists():
            _load_env_file(default_env)

    base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("MINIMAX_BASE_URL") or DEFAULT_MINIMAX_BASE_URL
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY") or ""
    if not base_url or not api_key:
        raise SystemExit("Missing env: set ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY (or MINIMAX_BASE_URL + MINIMAX_API_KEY).")

    done = _load_done_ids(out)

    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(inp):
        pid = str(obj.get("id") or "")
        if not pid or pid in done:
            continue
        rows.append(obj)
        if args.max_prompts and len(rows) >= int(args.max_prompts):
            break

    if not rows:
        print("[OK] Nothing to do (all done).")
        return 0

    random.Random(args.seed).shuffle(rows)

    def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        pid = str(item.get("id") or "")
        messages = item.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("bad messages")

        dialog = _format_dialog(messages)
        if not dialog:
            raise RuntimeError("empty dialog")

        user_prompt = (
            "下面是对话历史。请你作为“助手”回复最后一条用户消息，只输出回复正文。\n\n"
            f"{dialog}\n\n助手："
        )

        def _gen(cfg: GenCfg) -> str:
            return _with_retry(
                lambda: _call_anthropic_compat(
                    base_url=base_url,
                    api_key=api_key,
                    model=str(args.model),
                    system=str(args.system_assistant),
                    user_text=user_prompt,
                    max_tokens=512,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    timeout_s=int(args.timeout),
                    include_thinking=False,
                ),
                retries=int(args.retries),
                sleep_s=float(args.sleep),
                overload_wait_s=float(args.overload_wait),
            )

        a = _gen(GEN_A)
        b = _gen(GEN_B)

        judge_prompt = (
            "对话历史：\n"
            f"{dialog}\n\n"
            "候选回复 A：\n"
            f"{a}\n\n"
            "候选回复 B：\n"
            f"{b}\n\n"
            "请输出 JSON："
        )

        def _judge() -> Tuple[str, str]:
            raw = _with_retry(
                lambda: _call_anthropic_compat(
                    base_url=base_url,
                    api_key=api_key,
                    model=str(args.judge_model),
                    system=str(args.system_judge),
                    user_text=judge_prompt,
                    max_tokens=64,
                    temperature=0.0,
                    top_p=1.0,
                    timeout_s=int(args.timeout),
                    include_thinking=True,
                ),
                retries=int(args.retries),
                sleep_s=float(args.sleep),
                overload_wait_s=float(args.overload_wait),
            )
            try:
                return _parse_winner(raw), raw
            except Exception:
                raw2 = _with_retry(
                    lambda: _call_anthropic_compat(
                        base_url=base_url,
                        api_key=api_key,
                        model=str(args.judge_model),
                        system=SYSTEM_JUDGE_STRICT,
                        user_text=judge_prompt,
                        max_tokens=8,
                        temperature=0.0,
                        top_p=1.0,
                        timeout_s=int(args.timeout),
                        include_thinking=True,
                    ),
                    retries=max(1, int(args.retries) // 2),
                    sleep_s=float(args.sleep),
                    overload_wait_s=float(args.overload_wait),
                )
                try:
                    return _parse_winner(raw2), (raw + "\n\n--- retry_strict ---\n" + raw2)
                except Exception:
                    # Last resort: heuristic fallback so we still produce a pair.
                    sa = _simple_score(a)
                    sb = _simple_score(b)
                    winner = "A" if sa >= sb else "B"
                    dbg = (
                        raw
                        + "\n\n--- retry_strict ---\n"
                        + raw2
                        + f"\n\n--- fallback_score ---\nA={sa:.3f} B={sb:.3f} => {winner}"
                    )
                    return winner, dbg

        if args.dry_run:
            # Don't spend tokens; just return a preview object.
            return {
                "id": pid,
                "bucket": item.get("bucket"),
                "conversations": _to_sharegpt_conversations(messages),
                "chosen": {"from": "gpt", "value": a},
                "rejected": {"from": "gpt", "value": b},
                "meta": {"dry_run": True},
            }

        winner, judge_raw = _judge()

        chosen = a if winner == "A" else b
        rejected = b if winner == "A" else a

        return {
            "id": pid,
            "bucket": item.get("bucket"),
            "conversations": _to_sharegpt_conversations(messages),
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
            "meta": {
                "gen_model": str(args.model),
                "judge_model": str(args.judge_model),
                "gen_a": {"temperature": GEN_A.temperature, "top_p": GEN_A.top_p},
                "gen_b": {"temperature": GEN_B.temperature, "top_p": GEN_B.top_p},
                "judge_winner": winner,
                "judge_raw": judge_raw[:600],
            },
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, Any]] = []
    wrote = 0

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {ex.submit(_one, it): it for it in rows}
        for fut in as_completed(futs):
            it = futs[fut]
            pid = str(it.get("id") or "")
            try:
                obj = fut.result()
                _write_jsonl_append(out, [obj])
                wrote += 1
                if wrote % 50 == 0:
                    print(f"[OK] wrote={wrote} last_id={pid}")
            except Exception as e:
                failures.append({"id": pid, "error": str(e)[:500]})
                if len(failures) <= 10:
                    print(f"[FAIL] id={pid} err={e}")

    if failures:
        fail_path = out.with_suffix(out.suffix + ".failed.jsonl")
        _write_jsonl_append(fail_path, failures)
        print(f"[WARN] failures={len(failures)} written to {fail_path}")

    print("[DONE] output:", str(out))
    print("[DONE] wrote:", wrote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
