#!/usr/bin/env python3
"""Build intent_dataset_v4: USER-only windows of 1/2/3 turns, length-matched within turn buckets.

Online alignment:
  turn 1 -> last 1 USER
  turn 2 -> last 2 USER
  turn 3+ -> last 3 USER

No truncation: only keep windows whose natural char length falls in overlapping bins
between SoulChat (label=1) and LCCC (label=0).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


INTENT_INSTRUCTION = (
    "你是校园心理健康分流模型。根据多轮对话文本判断是否属于心理咨询/倾诉，只输出严格 JSON："
    '{"is_consult":0/1}。不要输出多余文本。'
)

RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_CAMPUS = re.compile(
    r"(学校|校园|大学|学院|老师|同学|室友|宿舍|寝室|宿管|舍友|班级|课表|课程|选课|补考|重修|"
    r"期中|期末|考试|挂科|绩点|GPA|作业|论文|开题|答辩|实验课|实验室|课题组|实习|面试|简历|"
    r"秋招|春招|校招|考研|保研|推免|导师|助教|辅导员|班主任|奖学金|竞赛|社团|学生会|图书馆|"
    r"自习室|教室|食堂|校医院|心理中心|校心理|宿舍楼|军训|学分|选专业|转专业|双学位|四六级|"
    r"英语|学习|课堂|出勤|点名)"
)
RE_STRONG_RISK = re.compile(r"(自杀|想死|割腕|跳楼|吞药|不想活了|结束生命|上吊|煤气)")


@dataclass
class Candidate:
    text: str
    n_turns: int
    chars: int
    label: int
    meta: Dict[str, Any]


@dataclass
class Sample:
    instruction: str
    input: str
    output: str
    meta: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "instruction": self.instruction,
                "input": self.input,
                "output": self.output,
                "meta": self.meta,
            },
            ensure_ascii=False,
        )


def _norm_text(text: str) -> str:
    text = (text or "").strip().replace("\u200b", "").replace("\ufeff", "")
    return text.replace(" ", "").strip()


def _is_campus_related(text: str) -> bool:
    return bool(RE_CAMPUS.search(_norm_text(text)))


def _is_ok_user_utterance(text: str, *, min_chars: int = 4, max_chars: int = 400) -> bool:
    t = _norm_text(text)
    if len(t) < min_chars or len(t) > max_chars:
        return False
    if not RE_CJK.search(t):
        return False
    if len(set(t)) / max(1, len(t)) < 0.12:
        return False
    if RE_STRONG_RISK.search(t):
        return False
    return True


def _format_user_window(turns: Sequence[str]) -> str:
    return "\n".join(f"[USER] {t}" for t in turns).strip()


def _iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        eof = False
        while True:
            if not buf and not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buf += chunk
                else:
                    eof = True
            buf = buf.lstrip()
            if buf.startswith("["):
                buf = buf[1:]
                break
            if eof:
                raise ValueError(f"Cannot find '[' in {path}")

        while True:
            while True:
                buf = buf.lstrip()
                if buf.startswith(","):
                    buf = buf[1:]
                    continue
                break
            if buf.lstrip().startswith("]"):
                return
            while True:
                try:
                    if buf.lstrip().startswith("]"):
                        return
                    obj, idx = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[idx:]
                    break
                except json.JSONDecodeError:
                    if eof:
                        return
                    chunk = f.read(chunk_size)
                    if chunk:
                        buf += chunk
                    else:
                        eof = True


def _write_jsonl(path: Path, rows: List[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.to_json() + "\n")


def _windows_from_user_turns(
    user_turns: List[str],
    *,
    rng: random.Random,
    label: int,
    source: str,
    dialog_key: str,
    max_windows_per_k: int = 4,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> List[Candidate]:
    out: List[Candidate] = []
    n = len(user_turns)
    if n <= 0:
        return out
    base_meta = dict(extra_meta or {})
    for k in (1, 2, 3):
        if n < k:
            continue
        starts = list(range(0, n - k + 1))
        rng.shuffle(starts)
        taken = 0
        for start in starts:
            if taken >= max_windows_per_k:
                break
            chunk = user_turns[start : start + k]
            if any(not _is_ok_user_utterance(u) for u in chunk):
                continue
            text = _format_user_window(chunk)
            # Natural length only; never truncate.
            if len(text) < 8:
                continue
            meta = {
                **base_meta,
                "source": source,
                "dialog_key": dialog_key,
                "n_turns": k,
                "window_start": start,
                "user_dialog_len": n,
                "campus_related": _is_campus_related(text),
                "builder": "intent_v4",
            }
            out.append(
                Candidate(
                    text=text,
                    n_turns=k,
                    chars=len(text),
                    label=label,
                    meta=meta,
                )
            )
            taken += 1
    return out


def _extract_soulchat_users(obj: dict) -> List[str]:
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return []
    users: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if (m.get("role") or "").strip() != "user":
            continue
        t = _norm_text(str(m.get("content") or ""))
        if t:
            users.append(t)
    return users


def _extract_lccc_users(dialog: Any) -> List[str]:
    """LCCC is a list of alternating utterances; even indices treated as USER."""
    if not isinstance(dialog, list):
        return []
    turns = [_norm_text(x) for x in dialog if isinstance(x, str)]
    turns = [t for t in turns if t]
    # Keep even indices as user side (same convention as common LCCC processing).
    return [turns[i] for i in range(0, len(turns), 2)]


def _collect_candidates(
    *,
    path: Path,
    source: str,
    label: int,
    rng: random.Random,
    max_scan: Optional[int],
    max_dialogs_keep_factor: float,
    extract_users,
) -> List[Candidate]:
    cands: List[Candidate] = []
    kept_dialogs = 0
    for idx, obj in enumerate(_iter_json_array(path)):
        if max_scan is not None and idx >= max_scan:
            break
        # Light random skip to diversify without full scan cost on huge files.
        if max_dialogs_keep_factor < 1.0 and rng.random() > max_dialogs_keep_factor:
            continue
        users = extract_users(obj)
        if len(users) < 1:
            continue
        dialog_key = f"{source}:{idx}"
        extra = {"idx": idx}
        if source == "SoulChatCorpus" and isinstance(obj, dict):
            extra["id"] = obj.get("id")
            extra["topic"] = obj.get("topic")
        cands.extend(
            _windows_from_user_turns(
                users,
                rng=rng,
                label=label,
                source=source,
                dialog_key=dialog_key,
                max_windows_per_k=3,
                extra_meta=extra,
            )
        )
        kept_dialogs += 1
    print(f"[{source}] scanned_upto={max_scan or 'all'} kept_dialogs~={kept_dialogs} candidates={len(cands)}")
    return cands


def _match_length_within_turns(
    pos: List[Candidate],
    neg: List[Candidate],
    *,
    n_pos: int,
    n_neg: int,
    bin_size: int,
    rng: random.Random,
    turn_weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[List[Candidate], List[Candidate], Dict[str, Any]]:
    """Match pos/neg inside each n_turns bucket by char-length bins (no truncation)."""
    assert abs(sum(turn_weights) - 3.0) < 1e-6 or abs(sum(turn_weights) - 1.0) < 1e-6
    weights = list(turn_weights)
    if abs(sum(weights) - 1.0) < 1e-6:
        weights = [w * 3.0 for w in weights]

    pos_by_k = {k: [c for c in pos if c.n_turns == k] for k in (1, 2, 3)}
    neg_by_k = {k: [c for c in neg if c.n_turns == k] for k in (1, 2, 3)}

    # Allocate quotas proportional to weights, then residual to largest remainders.
    raw_pos = [n_pos * (w / sum(weights)) for w in weights]
    raw_neg = [n_neg * (w / sum(weights)) for w in weights]
    quota_pos = [int(x) for x in raw_pos]
    quota_neg = [int(x) for x in raw_neg]
    while sum(quota_pos) < n_pos:
        frac = sorted([(raw_pos[i] - quota_pos[i], i) for i in range(3)], reverse=True)
        quota_pos[frac[0][1]] += 1
    while sum(quota_neg) < n_neg:
        frac = sorted([(raw_neg[i] - quota_neg[i], i) for i in range(3)], reverse=True)
        quota_neg[frac[0][1]] += 1

    picked_pos: List[Candidate] = []
    picked_neg: List[Candidate] = []
    match_report: Dict[str, Any] = {}

    for ki, k in enumerate((1, 2, 3)):
        need_p, need_n = quota_pos[ki], quota_neg[ki]
        pb: Dict[int, List[Candidate]] = defaultdict(list)
        nb: Dict[int, List[Candidate]] = defaultdict(list)
        for c in pos_by_k[k]:
            pb[c.chars // bin_size].append(c)
        for c in neg_by_k[k]:
            nb[c.chars // bin_size].append(c)
        for b in pb:
            rng.shuffle(pb[b])
        for b in nb:
            rng.shuffle(nb[b])

        bins = sorted(set(pb) & set(nb))
        rng.shuffle(bins)
        # Target equal take from each side per bin to keep length dist aligned.
        # Prefer filling the smaller quota side proportionally.
        take_p: List[Candidate] = []
        take_n: List[Candidate] = []
        # Multi-pass round robin over bins
        while (len(take_p) < need_p or len(take_n) < need_n) and bins:
            progressed = False
            next_bins = []
            for b in bins:
                if len(take_p) >= need_p and len(take_n) >= need_n:
                    break
                if not pb[b] or not nb[b]:
                    continue
                # Take one pair when both still need; else take for the side still needing
                # only if the other side already full (keeps leftover fill length-matched-ish).
                if len(take_p) < need_p and len(take_n) < need_n:
                    take_p.append(pb[b].pop())
                    take_n.append(nb[b].pop())
                    progressed = True
                elif len(take_p) < need_p and len(take_n) >= need_n and pb[b]:
                    # Only add pos if bin still has neg mass historically — skip to avoid skew.
                    pass
                elif len(take_n) < need_n and len(take_p) >= need_p and nb[b]:
                    pass
                if pb[b] and nb[b]:
                    next_bins.append(b)
            bins = next_bins
            if not progressed:
                break

        # If still short, pair-match greedily within remaining overlapping bins.
        if len(take_p) < need_p or len(take_n) < need_n:
            rem_bins = sorted(set(pb) & set(nb))
            rng.shuffle(rem_bins)
            for b in rem_bins:
                while pb[b] and nb[b] and (len(take_p) < need_p or len(take_n) < need_n):
                    if len(take_p) < need_p and len(take_n) < need_n:
                        take_p.append(pb[b].pop())
                        take_n.append(nb[b].pop())
                    else:
                        break

        match_report[str(k)] = {
            "quota_pos": need_p,
            "quota_neg": need_n,
            "got_pos": len(take_p),
            "got_neg": len(take_n),
            "overlap_bins": len(set(c.chars // bin_size for c in take_p) & set(c.chars // bin_size for c in take_n)),
            "pos_chars_mean": round(mean([c.chars for c in take_p]), 1) if take_p else 0,
            "neg_chars_mean": round(mean([c.chars for c in take_n]), 1) if take_n else 0,
            "pos_chars_med": sorted([c.chars for c in take_p])[len(take_p) // 2] if take_p else 0,
            "neg_chars_med": sorted([c.chars for c in take_n])[len(take_n) // 2] if take_n else 0,
        }
        if len(take_p) < need_p * 0.8 or len(take_n) < need_n * 0.8:
            print(
                f"[warn] turn={k} underfilled: pos {len(take_p)}/{need_p} neg {len(take_n)}/{need_n}"
            )
        picked_pos.extend(take_p)
        picked_neg.extend(take_n)

    rng.shuffle(picked_pos)
    rng.shuffle(picked_neg)
    return picked_pos[:n_pos], picked_neg[:n_neg], match_report


def _to_samples(cands: List[Candidate], split: str) -> List[Sample]:
    rows: List[Sample] = []
    for c in cands:
        meta = dict(c.meta)
        meta["split"] = split
        meta["chars"] = c.chars
        rows.append(
            Sample(
                instruction=INTENT_INSTRUCTION,
                input=c.text,
                output=json.dumps({"is_consult": int(c.label)}, ensure_ascii=False),
                meta=meta,
            )
        )
    return rows


def _stats(rows: List[Sample]) -> Dict[str, Any]:
    by_label = Counter()
    by_source = Counter()
    by_turns = Counter()
    lengths = []
    turns = []
    with_assistant = 0
    for row in rows:
        label = json.loads(row.output)["is_consult"]
        by_label[str(label)] += 1
        by_source[str(row.meta.get("source", "unknown"))] += 1
        nt = int(row.meta.get("n_turns") or row.input.count("[USER]"))
        by_turns[str(nt)] += 1
        turns.append(nt)
        lengths.append(len(row.input))
        with_assistant += int("[ASSISTANT]" in row.input)
    # length by label
    len_by_label: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        label = str(json.loads(row.output)["is_consult"])
        len_by_label[label].append(len(row.input))
    len_summary = {}
    for lab, xs in len_by_label.items():
        xs_sorted = sorted(xs)
        len_summary[lab] = {
            "mean": round(mean(xs), 1),
            "med": xs_sorted[len(xs_sorted) // 2],
            "p25": xs_sorted[len(xs_sorted) // 4],
            "p75": xs_sorted[(3 * len(xs_sorted)) // 4],
        }
    return {
        "count": len(rows),
        "by_label": dict(by_label),
        "by_source": dict(by_source),
        "by_n_turns": dict(by_turns),
        "avg_user_turns": round(mean(turns), 2) if turns else 0,
        "avg_text_len": round(mean(lengths), 1) if lengths else 0,
        "length_by_label": len_summary,
        "with_assistant_pct": round(100.0 * with_assistant / len(rows), 1) if rows else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="train/data/classifiers/train/intent")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--soulchat-path", default="archive/sources/soulchat/SoulChatCorpus-sft-multi-Turn.json")
    ap.add_argument("--lccc-dir", default="archive/sources/lccc")
    ap.add_argument("--bin-size", type=int, default=20)
    ap.add_argument("--consult-train", type=int, default=1500)
    ap.add_argument("--consult-valid", type=int, default=150)
    ap.add_argument("--consult-test", type=int, default=150)
    ap.add_argument("--chat-train", type=int, default=1500)
    ap.add_argument("--chat-valid", type=int, default=150)
    ap.add_argument("--chat-test", type=int, default=150)
    ap.add_argument("--soulchat-max-scan", type=int, default=80000)
    ap.add_argument("--lccc-train-max-scan", type=int, default=250000)
    ap.add_argument("--lccc-valid-max-scan", type=int, default=80000)
    ap.add_argument("--lccc-test-max-scan", type=int, default=80000)
    args = ap.parse_args()

    root = Path.cwd()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    soulchat_path = Path(args.soulchat_path)
    if not soulchat_path.is_absolute():
        soulchat_path = root / soulchat_path
    lccc_dir = Path(args.lccc_dir)
    if not lccc_dir.is_absolute():
        lccc_dir = root / lccc_dir

    rng = random.Random(args.seed)

    # Pool candidates (shared pools, then split by hash of dialog_key to avoid leakage across splits).
    pos_all = _collect_candidates(
        path=soulchat_path,
        source="SoulChatCorpus",
        label=1,
        rng=random.Random(args.seed + 1),
        max_scan=args.soulchat_max_scan,
        max_dialogs_keep_factor=1.0,
        extract_users=_extract_soulchat_users,
    )
    neg_train_pool = _collect_candidates(
        path=lccc_dir / "LCCC-base_train.json",
        source="LCCC-base",
        label=0,
        rng=random.Random(args.seed + 2),
        max_scan=args.lccc_train_max_scan,
        max_dialogs_keep_factor=0.35,
        extract_users=lambda d: _extract_lccc_users(d),
    )
    neg_valid_pool = _collect_candidates(
        path=lccc_dir / "LCCC-base_valid.json",
        source="LCCC-base",
        label=0,
        rng=random.Random(args.seed + 3),
        max_scan=args.lccc_valid_max_scan,
        max_dialogs_keep_factor=1.0,
        extract_users=lambda d: _extract_lccc_users(d),
    )
    neg_test_pool = _collect_candidates(
        path=lccc_dir / "LCCC-base_test.json",
        source="LCCC-base",
        label=0,
        rng=random.Random(args.seed + 4),
        max_scan=args.lccc_test_max_scan,
        max_dialogs_keep_factor=1.0,
        extract_users=lambda d: _extract_lccc_users(d),
    )

    def split_pos(cands: List[Candidate]) -> Dict[str, List[Candidate]]:
        import hashlib

        buckets: Dict[str, List[Candidate]] = {"train": [], "valid": [], "test": []}
        for c in cands:
            x = int(hashlib.md5(str(c.meta.get("dialog_key")).encode()).hexdigest()[:8], 16) % 1000
            if x < 800:
                buckets["train"].append(c)
            elif x < 900:
                buckets["valid"].append(c)
            else:
                buckets["test"].append(c)
        return buckets

    pos_split = split_pos(pos_all)
    neg_split = {
        "train": neg_train_pool,
        "valid": neg_valid_pool,
        "test": neg_test_pool,
    }

    sizes = {
        "train": (args.consult_train, args.chat_train),
        "valid": (args.consult_valid, args.chat_valid),
        "test": (args.consult_test, args.chat_test),
    }

    splits: Dict[str, List[Sample]] = {}
    match_reports = {}
    for split, (n_pos, n_neg) in sizes.items():
        p, n, report = _match_length_within_turns(
            pos_split[split],
            neg_split[split],
            n_pos=n_pos,
            n_neg=n_neg,
            bin_size=args.bin_size,
            rng=random.Random(args.seed + 100 + hash(split) % 97),
        )
        # Dedup within each side first, then top up from unused matched pool / original pools.
        def _unique(cands: List[Candidate]) -> List[Candidate]:
            seen_local = set()
            out = []
            for c in cands:
                if c.text in seen_local:
                    continue
                seen_local.add(c.text)
                out.append(c)
            return out

        uniq_p = _unique(p)
        uniq_n = _unique(n)
        used_texts = {c.text for c in uniq_p} | {c.text for c in uniq_n}

        def _top_up(have: List[Candidate], pool: List[Candidate], need: int, label: int) -> List[Candidate]:
            if len(have) >= need:
                return have[:need]
            # Prefer same-length-bin partners still available in pool.
            have_bins = {c.chars // args.bin_size for c in have}
            extras = [c for c in pool if c.text not in used_texts and c.label == label]
            rng.shuffle(extras)
            extras.sort(key=lambda c: 0 if (c.chars // args.bin_size) in have_bins else 1)
            for c in extras:
                if len(have) >= need:
                    break
                have.append(c)
                used_texts.add(c.text)
            if len(have) < need:
                raise RuntimeError(
                    f"[{split}] cannot fill label={label}: got={len(have)} need={need}"
                )
            return have[:need]

        uniq_p = _top_up(uniq_p, pos_split[split], n_pos, 1)
        uniq_n = _top_up(uniq_n, neg_split[split], n_neg, 0)
        if len(uniq_p) != n_pos or len(uniq_n) != n_neg:
            raise RuntimeError(
                f"[{split}] size mismatch after top-up: pos={len(uniq_p)}/{n_pos} neg={len(uniq_n)}/{n_neg}"
            )
        rows = _to_samples(uniq_p, split) + _to_samples(uniq_n, split)
        rng.shuffle(rows)
        assert len(rows) == n_pos + n_neg
        splits[split] = rows
        match_reports[split] = report
        _write_jsonl(out_dir / f"intent_{split}.jsonl", rows)
        print(f"[{split}] wrote {len(rows)} (pos={len(uniq_p)} neg={len(uniq_n)})")

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "notes": [
            "USER-only inputs with [USER] prefixes; no [ASSISTANT].",
            "Windows are 1/2/3 USER turns sampled at random positions in each dialog.",
            "Length matched within the same n_turns bucket via char bins; no truncation.",
            "Labels remain source-level: SoulChat=1, LCCC=0.",
            "Aligned with online rule: 1st->1, 2nd->2, 3+->last 3 USER turns.",
        ],
        "splits": {s: _stats(rows) for s, rows in splits.items()},
        "match_reports": match_reports,
        "config": {
            "bin_size": args.bin_size,
            "consult_sizes": {
                "train": args.consult_train,
                "valid": args.consult_valid,
                "test": args.consult_test,
            },
            "chat_sizes": {
                "train": args.chat_train,
                "valid": args.chat_valid,
                "test": args.chat_test,
            },
            "soulchat_path": str(soulchat_path),
            "lccc_dir": str(lccc_dir),
            "soulchat_max_scan": args.soulchat_max_scan,
            "lccc_train_max_scan": args.lccc_train_max_scan,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"splits": stats["splits"], "match_reports": match_reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
