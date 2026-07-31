# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/add_rejected_minimax.py
# 原先用途: 为已有 prompt/chosen 用 MiniMax 等补全 rejected，扩充 DPO 对。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


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
                did = str(obj.get("id") or "")
                if did:
                    done.add(did)
            except Exception:
                continue
    return done


def _load_env_file(path: Path) -> None:
    """
    Load simple KEY=VALUE env file (no python-dotenv dependency).
    - ignores empty lines and # comments
    - does NOT override existing os.environ
    - supports a single-line file containing only the API key
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            if line.startswith("http://") or line.startswith("https://"):
                os.environ.setdefault("ANTHROPIC_BASE_URL", line)
            else:
                # Support a single-line file that only contains an API key.
                # Do not assume key prefix (not all providers use "sk-").
                os.environ.setdefault("ANTHROPIC_API_KEY", line)
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
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
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    timeout_s: int,
    include_thinking: bool = False,
) -> Tuple[str, str]:
    url = _anthropic_messages_url(base_url)
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system or "",
    }
    if messages is not None:
        payload["messages"] = messages
    else:
        payload["messages"] = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    if top_p is not None:
        payload["top_p"] = float(top_p)
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    parts: List[str] = []
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    content = data.get("content")
    if isinstance(content, list):
        for blk in content:
            if not isinstance(blk, dict):
                continue
            t = str(blk.get("type") or "")
            if t == "text":
                seg = str(blk.get("text") or blk.get("content") or blk.get("value") or "")
                if seg:
                    text_parts.append(seg)
                    parts.append(seg)
            elif include_thinking and t == "thinking":
                parts.append(str(blk.get("thinking") or blk.get("text") or blk.get("content") or ""))
            elif t == "thinking":
                seg = str(blk.get("thinking") or blk.get("text") or blk.get("content") or "")
                if seg:
                    thinking_parts.append(seg)
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
    text_out = "\n".join([p for p in text_parts if p]).strip()
    thinking_out = "\n".join([p for p in thinking_parts if p]).strip()
    # Prefer returning only text blocks for training; thinking is returned separately for debugging.
    if text_out:
        return text_out, thinking_out
    # If there is no structured text block, fall back to whatever we can find.
    return "\n".join([p for p in parts if p]).strip(), thinking_out


def _with_retry(fn, *, retries: int, sleep_s: float, overload_wait_s: float) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e)
            wait = sleep_s * (2**i)
            if "HTTP 529" in msg or "overloaded" in msg.lower():
                wait = max(wait, overload_wait_s)
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {last_err}")


def _norm(t: str) -> str:
    return "".join((t or "").split()).lower()


_RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]", re.UNICODE)
_RE_CJK = re.compile(r"[\u4e00-\u9fff]")


def _strip_emoji(text: str) -> str:
    return _RE_EMOJI.sub("", text or "")


def _strip_wrapping_quotes(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    pairs = [
        ('"', '"'),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
    ]
    for l, r in pairs:
        if t.startswith(l) and t.endswith(r) and len(t) >= 2:
            t = t[1:-1].strip()
            break
    return t


def _postprocess_candidate(text: str) -> str:
    """
    Gateways sometimes output meta + an example like:
      比如：
      "……真正的回复……"
    Try to recover the actual reply.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # If it contains an explicit example prefix, try to take the quoted content or the last line.
    if any(x in t for x in ("比如", "例如", "示例")):
        q1 = t.find('"')
        if q1 != -1:
            t2 = t[q1:].strip()
            t2 = _strip_wrapping_quotes(t2)
            if t2:
                t = t2
        parts = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if parts:
            t = parts[-1]

    # Clean up prefixes that look like thoughts or label prefixes
    t = re.sub(r"^(?:分析|思考|思路|思考过程).*?(?:回复|回答|建议)[：:]", "", t, flags=re.DOTALL)
    t = re.sub(r"^\s*(?:回复|回答|建议)\s*[:：]\s*", "", t)
    t = re.sub(r"^\s*[-•]\s*", "", t)

    t = _strip_wrapping_quotes(t)
    t = t.strip()
    return t


def _format_conversations(convs: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in convs:
        if not isinstance(m, dict):
            continue
        fr = str(m.get("from") or "").strip().lower()
        val = str(m.get("value") or "").strip()
        if not val:
            continue
        if fr == "human":
            lines.append(f"用户：{val}")
        elif fr == "gpt":
            lines.append(f"助手：{val}")
    return "\n".join(lines).strip()


SYSTEM_BAD_REPLY_ZH = (
    "我正在为一个心理咨询数据集构造 DPO 的 rejected 样本。\n"
    "请你对用户最后一条心理咨询消息给出“较差但仍相关”的中文回复：\n"
    "- 不够共情、较空泛敷衍、承接不足、略显复读，或者是显得有点说教等；\n"
    "输出要求：你可以先进行简短的思考，但最终的回复正文必须包裹在 <reply> 和 </reply> 标签中。除了 <reply> 标签内的内容外，不要解释原因。\n"
    "安全要求：不要热线/电话；不要提供自伤自杀的方法、步骤、工具、剂量或可操作细节。\n"
    "长度：<reply> 标签内的正文应为 30-100 字，1-3 句。"
)


def _make_user_text(convs: List[Dict[str, str]]) -> str:
    dialog = _format_conversations(convs)
    return (
        f"{dialog}\n\n"
        "请作为助手回复最后一条用户消息，只输出回复正文。第一字符必须是中文汉字。"
    )


def _to_anthropic_messages(convs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for m in convs:
        if not isinstance(m, dict):
            continue
        fr = str(m.get("from") or "").strip().lower()
        val = str(m.get("value") or "").strip()
        if not val:
            continue
        if fr == "human":
            msgs.append({"role": "user", "content": [{"type": "text", "text": val}]})
        elif fr == "gpt":
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": val}]})
    # Ensure ends with a user message (we reply to the last user turn).
    while msgs and msgs[-1]["role"] != "user":
        msgs.pop()
    return msgs


_RE_BAD_META = re.compile(r"(the user|assistant:|the assistant)", re.IGNORECASE)
_RE_BAD_META_ZH = re.compile(
    r"(比如|示例|这是|要求|质量|我们可以|作为\\s*rejected|输出格式|严格只用|只用如下格式|用户\\s*说|用户：|助手：|用户|助手|对话|主题相关|清晰步骤|可执行建议|不给出)",
    re.IGNORECASE,
)
_RE_QUOTE = re.compile(r"[\"“”‘’]")
_RE_PLACEHOLDER = re.compile(r"(\[reply\s*content\]|reply\s*content|在这里写|\(在这里写|（在这里写)", re.IGNORECASE)
_RE_ASCII_WORD = re.compile(r"[A-Za-z]{4,}")
_RE_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)
_RE_MARKED = re.compile(r"<<<\s*REJECTED\s*>>>\s*(.*?)\s*<<<\s*END\s*>>>", re.DOTALL | re.IGNORECASE)


_RE_THINKING = re.compile(r"<(think|thinking|thought|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

_RE_MULTI_MARKERS = re.compile(r"<(reply|rejected)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE)

def _extract_rejected(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    
    # Strip any XML-like thinking tags that might have leaked into the text block
    t = _RE_THINKING.sub("", t).strip()

    # 1) If <reply> tags exist, strictly prefer them
    m_reply = _RE_MULTI_MARKERS.search(t)
    if m_reply:
        return m_reply.group(2).strip()

    # 2) Fallback: if we reached here, there is NO <reply> tag at all.
    # The output might be a pure English reasoning block like "The user asks for..."
    # Since we strictly told it to put the final text in <reply>, if it didn't, 
    # the entire text is likely just a preamble/reasoning that failed to finalize.
    # In order to not pollute the dataset with garbage English reasoning, we try to 
    # salvage Chinese sections, or just return empty string to mark it as failed.
    
    # Only keep lines that have a reasonable amount of CJK characters, skipping pure English reasoning lines.
    cleaned: List[str] = []
    for line in t.replace("```", "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper() in ("<<<REJECTED>>>", "<<<END>>>") or s.startswith("答：") or s.startswith("回复："):
            s = re.sub(r'^(<<<REJECTED>>>|答：|回复：|回复内容：|<<<END>>>)\s*', '', s, flags=re.IGNORECASE)
            
        if not s:
            continue
            
        # Ignore purely english thinking/reasoning blocks
        if len(_RE_CJK.findall(s)) < 5 and len(re.findall(r'[a-zA-Z]', s)) > 10:
            continue 
            
        cleaned.append(s)
    if not cleaned:
        return ""
    
    # Sort cleaned parts by their score: length of strings that are predominantly CJK
    def cjk_score(s: str) -> int:
        return len(_RE_CJK.findall(s)) * 2 - len(re.findall(r'[a-zA-Z]', s))
        
    best = max(cleaned, key=cjk_score)
    if cjk_score(best) <= 0:
        return "" # If it's mostly English reasoning 
        
    if len(_RE_CJK.findall(best)) < 3: 
        return "" # Not a valid Chinese reply candidate
        
    if len(best) >= 5:
        return best.strip()
    return "\n".join(cleaned).strip()


def _truncate(s: str, *, max_chars: int) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


def _build_one(
    obj: Dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    timeout_s: int,
    retries: int,
    sleep_s: float,
    overload_wait_s: float,
) -> Dict[str, Any]:
    row_id = str(obj.get("id") or "")
    convs = obj.get("conversations")
    if not row_id or not isinstance(convs, list) or not convs:
        raise ValueError("Bad row: missing id/conversations")

    chosen_obj = obj.get("chosen") or {}
    chosen_text = ""
    if isinstance(chosen_obj, dict):
        chosen_text = str(chosen_obj.get("value") or "")
    elif isinstance(chosen_obj, str):
        chosen_text = chosen_obj
    chosen_text = chosen_text.strip()
    if not chosen_text:
        raise ValueError(f"Bad row: missing chosen (id={row_id})")

    messages_payload = _to_anthropic_messages(convs)
    if not messages_payload:
        raise ValueError(f"Bad row: empty messages (id={row_id})")
    last_user = ""
    for m in reversed(convs):
        if not isinstance(m, dict):
            continue
        fr = str(m.get("from") or "").strip().lower()
        if fr == "human":
            last_user = str(m.get("value") or "").strip()
            if last_user:
                break

    # Single call per row: if invalid, mark as failed rather than fallback.
    # MiniMax Anthropic-compatible API may emit a thinking block before the text.
    # Use a sufficiently large max_tokens so the final text block is not truncated away.
    tok = min(int(max_tokens), 2048)
    raw_text, raw_thinking = _with_retry(
        lambda: _call_anthropic_compat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system=SYSTEM_BAD_REPLY_ZH,
            user_text="",
            messages=messages_payload,
            max_tokens=tok,
            temperature=float(temperature),
            top_p=top_p,
            timeout_s=timeout_s,
            include_thinking=False,
        ),
        retries=retries,
        sleep_s=sleep_s,
        overload_wait_s=overload_wait_s,
    )

    cand = _extract_rejected(raw_text)
    cand = _strip_emoji(cand).strip()
    cand = _postprocess_candidate(cand)

    out = dict(obj)
    out["rejected"] = {"from": "gpt", "value": cand}
    meta = out.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        out["meta"] = meta
    flags: List[str] = []
    if not cand:
        flags.append("empty_text")
    if cand and _norm(cand) == _norm(chosen_text):
        flags.append("equals_chosen")
    if cand and last_user:
        nu = _norm(last_user)
        nc = _norm(cand)
        if nu and nc and (nc in nu or nu in nc):
            flags.append("echoes_user")
    if cand and len(_norm(cand)) < 10:
        flags.append("too_short")

    meta.update({"rejected_gen_model": model, "rejected_gen_temperature": float(temperature)})
    meta.update(
        {
            "rejected_gen_used_fallback": False,
            "rejected_gen_flags": flags,
            "rejected_gen_is_usable": (len(flags) == 0),
        }
    )
    # Store raw separated outputs to help inspect issues without discarding samples.
    meta.update(
        {
            "rejected_gen_raw_text": _truncate(raw_text, max_chars=1200),
            "rejected_gen_raw_thinking": _truncate(raw_thinking, max_chars=1200),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Add rejected replies using MiniMax (Anthropic-compatible) to chosen-only JSONL.")
    ap.add_argument("--input", required=True, help="Input JSONL with conversations + chosen.")
    ap.add_argument("--output", required=True, help="Output JSONL with conversations + chosen + rejected.")
    ap.add_argument(
        "--env-file",
        default="",
        help="Optional env file path containing ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY.",
    )
    ap.add_argument("--model", default="MiniMax-M2.7")
    ap.add_argument("--max-rows", type=int, default=0, help="Limit rows to process (0 = all).")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Append to output every N completed rows (1 = safest for interruption; higher = faster).",
    )
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--overload-wait", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Resume from existing output by skipping done ids.")
    args = ap.parse_args()

    if args.env_file:
        _load_env_file(Path(args.env_file))

    base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("MINIMAX_BASE_URL") or "https://api.minimaxi.com/anthropic"
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY") or ""
    if not api_key:
        raise SystemExit("Missing env: set ANTHROPIC_API_KEY (or MINIMAX_API_KEY), or pass --env-file.")

    rng = random.Random(int(args.seed))

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    done_ids: set[str] = set()
    if args.resume:
        done_ids = _load_done_ids(out)
        if done_ids:
            print(f"[INFO] resume: loaded {len(done_ids)} done ids from {out}")

    rows = list(_iter_jsonl(inp))
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]

    # Shuffle to avoid topic clustering in case of partial runs.
    rng.shuffle(rows)

    pending = [r for r in rows if str(r.get("id") or "") and str(r.get("id") or "") not in done_ids]
    if not pending:
        print("[OK] nothing to do.", flush=True)
        return 0

    print(f"[INFO] pending={len(pending)} workers={int(args.workers)} model={args.model}", flush=True)

    buf: List[Dict[str, Any]] = []
    flush_every = max(1, int(args.flush_every))
    wrote = 0

    failed = 0
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {
            ex.submit(
                _build_one,
                r,
                base_url=base_url,
                api_key=api_key,
                model=str(args.model),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p) if args.top_p is not None else None,
                timeout_s=int(args.timeout),
                retries=int(args.retries),
                sleep_s=float(args.sleep),
                overload_wait_s=float(args.overload_wait),
            ): r
            for r in pending
        }

        for fut in as_completed(futs):
            src = futs[fut]
            try:
                out_row = fut.result()
            except Exception as e:
                failed += 1
                print(f"[FAIL] {e}", flush=True)
                continue

            buf.append(out_row)
            if len(buf) >= flush_every:
                _write_jsonl_append(out, buf)
                wrote += len(buf)
                buf.clear()
                print(f"[INFO] wrote {wrote}", flush=True)

    if buf:
        _write_jsonl_append(out, buf)
        wrote += len(buf)
        buf.clear()

    print(f"[OK] wrote {wrote} -> {out} (failed={failed})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
