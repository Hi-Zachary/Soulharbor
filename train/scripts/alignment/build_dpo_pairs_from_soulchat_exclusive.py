# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/build_dpo_pairs_from_soulchat_exclusive.py
# 原先用途: 从 SoulChat 独占语料构造 DPO 偏好对。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _norm(t: str) -> str:
    return "".join((t or "").split()).lower()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


_RE_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]", re.UNICODE)
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _clean_assistant_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*(?:题主|楼主|谢邀|谢邀。|谢邀，)\s*[:：,，]?\s*", "", t)
    t = _RE_EMOJI.sub("", t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    return t.strip()


def _first_sentence(text: str, *, min_chars: int = 30, max_chars: int = 120) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[。！？!?\n])", t)
    out = ""
    for p in parts:
        if not p.strip():
            continue
        if len(out) + len(p) > max_chars and len(out) >= min_chars:
            break
        out += p
        if len(out) >= min_chars:
            break
    out = out.strip()
    if not out:
        out = t[:max_chars].strip()
    return out


GENERIC_REJECTS = [
    "我理解你的感受。你先别想太多，慢慢来。",
    "听起来你最近挺不容易的。先休息一下，之后再说。",
    "我明白你的困扰。你可以先放松一下，看看会不会好些。",
    "这种情况很常见，不用太担心。",
]


def _signature_from_conversations(convs: List[Dict[str, str]], *, max_msgs: int) -> str:
    """
    Build a fuzzy signature for dedup / exclusion.
    Only uses the first `max_msgs` turns to keep it stable across packaging.
    """
    parts: List[str] = []
    for m in convs[:max_msgs]:
        if not isinstance(m, dict):
            continue
        fr = str(m.get("from") or "").strip().lower()
        val = str(m.get("value") or "").strip()
        if not val:
            continue
        parts.append(f"{fr}:{_norm(val)}")
    return _sha1("|".join(parts))


def _signature_from_messages(msgs: List[Dict[str, str]], *, max_msgs: int) -> str:
    parts: List[str] = []
    for m in msgs[:max_msgs]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:{_norm(content)}")
    return _sha1("|".join(parts))


def _load_exclude_signatures(path: Path, *, sig_max_msgs: int) -> set[str]:
    """
    Exclude SFT-used SoulChat samples.
    `archive/llm_build/sft_soulchat.jsonl` does not include SoulChatCorpus ids,
    so we exclude by a content signature of the first few turns.
    """
    sigs: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            convs = obj.get("conversations")
            if not isinstance(convs, list) or not convs:
                continue
            sigs.add(_signature_from_conversations(convs, max_msgs=sig_max_msgs))
    return sigs


@dataclass(frozen=True)
class Candidate:
    topic: str
    prompt_convs: List[Dict[str, str]]  # must end with human
    chosen: str


def _build_candidates_from_soulchat(
    soulchat: List[Dict[str, Any]],
    *,
    exclude_sigs: set[str],
    sig_max_msgs: int,
    only_first_k_pairs: int,
    also_take_pair_n: int,
    max_prompt_msgs: int,
    min_chosen_chars: int,
    max_prompt_chars: int,
    max_chosen_chars: int,
    max_total_chars: int,
) -> Tuple[Dict[str, List[Candidate]], List[str]]:
    by_topic: Dict[str, List[Candidate]] = defaultdict(list)
    chosen_pool: List[str] = []

    for ex in soulchat:
        if not isinstance(ex, dict):
            continue
        topic = str(ex.get("topic") or "other").strip() or "other"
        msgs = ex.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        sig = _signature_from_messages(msgs, max_msgs=sig_max_msgs)
        if sig in exclude_sigs:
            continue

        # Collect all user->assistant pairs indices (by user message index).
        pair_user_idxs: List[int] = []
        for i in range(len(msgs) - 1):
            m_user = msgs[i]
            m_asst = msgs[i + 1]
            if not isinstance(m_user, dict) or not isinstance(m_asst, dict):
                continue
            if str(m_user.get("role") or "") != "user":
                continue
            if str(m_asst.get("role") or "") != "assistant":
                continue
            pair_user_idxs.append(i)

        # Select timepoints: first K pairs + optionally also the Nth pair (1-based) later in the convo.
        selected_user_idxs: List[int] = []
        k = max(0, int(only_first_k_pairs))
        selected_user_idxs.extend(pair_user_idxs[:k])
        n = int(also_take_pair_n)
        if n > 0 and n <= len(pair_user_idxs):
            selected_user_idxs.append(pair_user_idxs[n - 1])
        # Dedup & keep order.
        seen_idx: set[int] = set()
        selected_user_idxs = [x for x in selected_user_idxs if (x not in seen_idx and not seen_idx.add(x))]

        for user_i in selected_user_idxs:
            m_asst = msgs[user_i + 1]
            chosen = _clean_assistant_text(str(m_asst.get("content") or ""))
            if len(_norm(chosen)) < min_chosen_chars:
                continue
            if max_chosen_chars > 0 and len(chosen) > max_chosen_chars:
                continue

            # Prompt is a contiguous prefix ending at this user message.
            prompt_prefix = msgs[: user_i + 1]
            # If the prefix is too long, keep the *tail* so the end user query is preserved.
            if max_prompt_msgs > 0 and len(prompt_prefix) > max_prompt_msgs:
                prompt_prefix = prompt_prefix[-max_prompt_msgs:]

            convs: List[Dict[str, str]] = []
            for m in prompt_prefix:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "")
                content = str(m.get("content") or "")
                if not content.strip():
                    continue
                if role == "user":
                    convs.append({"from": "human", "value": content})
                elif role == "assistant":
                    convs.append({"from": "gpt", "value": _clean_assistant_text(content)})

            # Ensure prompt ends with human (the target user at this timepoint).
            while convs and convs[-1]["from"] != "human":
                convs.pop()
            if not convs:
                continue

            prompt_text = "\n".join([m.get("value", "") for m in convs if isinstance(m, dict)])
            if max_prompt_chars > 0 and len(prompt_text) > max_prompt_chars:
                continue
            if max_total_chars > 0 and (len(prompt_text) + len(chosen)) > max_total_chars:
                continue

            by_topic[topic].append(Candidate(topic=topic, prompt_convs=convs, chosen=chosen))
            chosen_pool.append(chosen)

    return by_topic, chosen_pool


def _pick_mismatch(pool: List[str], avoid: str, *, rng: random.Random, max_tries: int = 20) -> Optional[str]:
    if len(pool) < 2:
        return None
    for _ in range(max_tries):
        cand = pool[rng.randrange(0, len(pool))]
        if _norm(cand) == _norm(avoid):
            continue
        return cand
    return None


def _hash_row(conversations: List[Dict[str, str]], chosen: str, rejected: str) -> str:
    blob = json.dumps(
        {"conversations": conversations, "chosen": chosen, "rejected": rejected},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _hash_row_chosen_only(conversations: List[Dict[str, str]], chosen: str) -> str:
    blob = json.dumps(
        {"conversations": conversations, "chosen": chosen},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build DPO pairs from SoulChatCorpus, excluding SFT-used samples, focusing on early turns."
    )
    ap.add_argument(
        "--soulchat",
        default="archive/sources/soulchat/SoulChatCorpus-sft-multi-Turn.json",
        help="SoulChatCorpus JSON array path.",
    )
    ap.add_argument(
        "--exclude-sft",
        default="archive/llm_build/sft_soulchat.jsonl",
        help="SFT dataset built from SoulChat (used for exclusion by signature).",
    )
    ap.add_argument("--output", required=True, help="Output DPO JSONL (LLaMA-Factory ShareGPT DPO style).")
    ap.add_argument(
        "--output-mode",
        choices=["dpo", "chosen_only"],
        default="dpo",
        help="dpo: write chosen+rejected pairs; chosen_only: write (prompt, chosen) only for later augmentation.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target-total",
        type=int,
        default=30000,
        help="Target number of DPO rows (topic-balanced round-robin).",
    )
    ap.add_argument(
        "--only-first-k-pairs",
        type=int,
        default=2,
        help="Only use the first K user->assistant pairs of each conversation to avoid missing context.",
    )
    ap.add_argument(
        "--also-take-pair-n",
        type=int,
        default=0,
        help="Also take the Nth user->assistant pair (1-based) as an extra later timepoint (0 = disabled).",
    )
    ap.add_argument(
        "--max-prompt-msgs",
        type=int,
        default=12,
        help="Max number of messages to keep in the prompt prefix (from the beginning).",
    )
    ap.add_argument(
        "--max-prompt-chars",
        type=int,
        default=0,
        help="Optional hard cap on prompt characters after formatting (0 = disabled).",
    )
    ap.add_argument(
        "--max-chosen-chars",
        type=int,
        default=0,
        help="Optional hard cap on chosen characters after cleanup (0 = disabled).",
    )
    ap.add_argument(
        "--max-total-chars",
        type=int,
        default=0,
        help="Optional hard cap on (prompt_chars + chosen_chars) (0 = disabled).",
    )
    ap.add_argument("--min-chosen-chars", type=int, default=60, help="Skip chosen answers shorter than this.")
    ap.add_argument(
        "--sig-max-msgs",
        type=int,
        default=6,
        help="How many initial turns to include in the exclusion signature.",
    )
    ap.add_argument(
        "--reject-mode",
        choices=["mix", "mismatch", "truncate", "generic"],
        default="mix",
        help="How to generate rejected replies.",
    )
    ap.add_argument("--mismatch-prob", type=float, default=0.7, help="When reject-mode=mix, prob of mismatch.")
    ap.add_argument("--truncate-max-chars", type=int, default=120, help="When truncate is used, keep <= N chars.")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    soulchat_path = Path(args.soulchat)
    if not soulchat_path.exists():
        raise SystemExit(f"Missing SoulChatCorpus: {soulchat_path}")

    exclude_path = Path(args.exclude_sft)
    exclude_sigs: set[str] = set()
    if exclude_path.exists():
        exclude_sigs = _load_exclude_signatures(exclude_path, sig_max_msgs=int(args.sig_max_msgs))
        print(f"[INFO] loaded exclude signatures: {len(exclude_sigs)} from {exclude_path}")
    else:
        print(f"[WARN] exclude file missing, will not exclude SFT data: {exclude_path}")

    print(f"[INFO] loading SoulChatCorpus JSON: {soulchat_path}")
    with soulchat_path.open("r", encoding="utf-8") as f:
        soulchat = json.load(f)
    if not isinstance(soulchat, list):
        raise SystemExit("SoulChatCorpus must be a JSON array.")

    by_topic, chosen_pool = _build_candidates_from_soulchat(
        soulchat,
        exclude_sigs=exclude_sigs,
        sig_max_msgs=int(args.sig_max_msgs),
        only_first_k_pairs=int(args.only_first_k_pairs),
        also_take_pair_n=int(args.also_take_pair_n),
        max_prompt_msgs=int(args.max_prompt_msgs),
        min_chosen_chars=int(args.min_chosen_chars),
        max_prompt_chars=int(args.max_prompt_chars),
        max_chosen_chars=int(args.max_chosen_chars),
        max_total_chars=int(args.max_total_chars),
    )
    topics = sorted(by_topic.keys())
    total_candidates = sum(len(v) for v in by_topic.values())
    print(f"[INFO] candidates: {total_candidates} across {len(topics)} topics")

    if total_candidates == 0:
        raise SystemExit("No candidates after exclusion/filters. Try increasing only-first-k-pairs or lowering min-chosen-chars.")

    # Shuffle per topic so round-robin is randomized but deterministic via seed.
    for t in topics:
        rng.shuffle(by_topic[t])

    # Round-robin fill to target total to keep topic balance (rare topics get a chance).
    target = int(args.target_total)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    idx = {t: 0 for t in topics}
    with out_path.open("w", encoding="utf-8") as f:
        while wrote < target:
            progressed = False
            for t in topics:
                j = idx[t]
                if j >= len(by_topic[t]):
                    continue
                cand = by_topic[t][j]
                idx[t] = j + 1
                progressed = True

                chosen = cand.chosen
                out_mode = str(args.output_mode)
                rejected: Optional[str] = None
                use_mode: Optional[str] = None
                if out_mode == "dpo":
                    mode = str(args.reject_mode)
                    use_mode = mode
                    if mode == "mix":
                        use_mode = "mismatch" if (rng.random() < float(args.mismatch_prob)) else "truncate"

                    if use_mode == "mismatch":
                        rejected = _pick_mismatch(chosen_pool, chosen, rng=rng)
                        rejected = _clean_assistant_text(rejected or "")
                    elif use_mode == "truncate":
                        rejected = _first_sentence(chosen, max_chars=int(args.truncate_max_chars))
                    elif use_mode == "generic":
                        rejected = GENERIC_REJECTS[rng.randrange(0, len(GENERIC_REJECTS))]

                    rejected = (rejected or "").strip()
                    if not rejected or _norm(rejected) == _norm(chosen):
                        rejected = _first_sentence(chosen, max_chars=int(args.truncate_max_chars)).strip()
                    if not rejected or _norm(rejected) == _norm(chosen):
                        continue

                if out_mode == "dpo":
                    row_id = _hash_row(cand.prompt_convs, chosen, str(rejected or ""))
                else:
                    row_id = _hash_row_chosen_only(cand.prompt_convs, chosen)
                f.write(
                    json.dumps(
                        {
                            "id": row_id,
                            "bucket": t,
                            "conversations": cand.prompt_convs,
                            "chosen": {"from": "gpt", "value": chosen},
                            **({"rejected": {"from": "gpt", "value": rejected}} if out_mode == "dpo" else {}),
                            "meta": {
                                "source": str(soulchat_path),
                                "excluded_sft": str(exclude_path) if exclude_sigs else None,
                                "sig_max_msgs": int(args.sig_max_msgs),
                                "only_first_k_pairs": int(args.only_first_k_pairs),
                                "max_prompt_msgs": int(args.max_prompt_msgs),
                                **({"reject_mode": use_mode} if out_mode == "dpo" else {}),
                                "topic": t,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                wrote += 1
                if wrote >= target:
                    break

            if not progressed:
                break

    print(f"[OK] wrote {wrote} rows -> {out_path}")
    if wrote < target:
        print(f"[WARN] target_total={target} but only wrote {wrote} (ran out of candidates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
