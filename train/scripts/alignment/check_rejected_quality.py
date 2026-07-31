# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/alignment/check_rejected_quality.py
# 原先用途: 检查 DPO rejected 回复质量，过滤劣质负样本。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_ASCII = re.compile(r"[A-Za-z]")

# English reasoning / prompt echo
_RE_EN_META = re.compile(
    r"(the user|system prompt|assistant:|analysis|reasoning|thinking|thought process|"
    r"you are a helpful assistant|we need to)",
    re.IGNORECASE,
)

# Extra English meta patterns observed in some generations
_RE_EN_META_EXTRA = re.compile(
    r"(now count|let's count|so total|characters?\s*[:=]|within\s+\d+|that's within|fine\.)",
    re.IGNORECASE,
)

# Chinese meta markers (avoid matching normal phrases like “换位思考”)
_RE_ZH_META = re.compile(
    r"(根据要求|系统提示|输出格式|不要输出|只输出|"
    r"(思考|分析|推理)(过程|如下|：|:)|"
    r"我需要(思考|分析|推理)|让我(思考|分析|推理)|"
    r"下面(是|进行).{0,10}(思考|分析|推理))",
    re.IGNORECASE,
)

# Any XML-ish tags that commonly wrap thinking or replies
_RE_XML_TAGS = re.compile(
    r"</?(reply|rejected|think|thinking|reasoning|analysis|回复)[^>]*>",
    re.IGNORECASE,
)

# Generic angle-bracket tags (catch malformed wrappers too)
_RE_ANY_TAG = re.compile(r"<[^>]{1,30}>")
_RE_DQUOTE = re.compile(r"\"")


def _count_cjk(s: str) -> int:
    return len(_RE_CJK.findall(s or ""))


def _count_ascii(s: str) -> int:
    return len(_RE_ASCII.findall(s or ""))


def _snippet(s: str, n: int = 160) -> str:
    t = (s or "").replace("\n", " ").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


@dataclass(frozen=True)
class BadCase:
    row_id: str
    bucket: str
    flags: List[str]
    rejected_snip: str


def _get_rejected_value(obj: Dict[str, Any]) -> str:
    rj = obj.get("rejected")
    if isinstance(rj, dict):
        return str(rj.get("value") or "")
    if isinstance(rj, str):
        return rj
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Check rejected.value quality in a DPO JSONL.")
    ap.add_argument("--input", required=True, help="Input JSONL path.")
    ap.add_argument("--max-examples", type=int, default=20, help="Max examples to print per flag.")
    ap.add_argument("--write-bad", default="", help="If set, write bad rows to this JSONL (original row + flags).")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    by_flag: Counter[str] = Counter()
    bad_rows: List[Dict[str, Any]] = []
    examples: Dict[str, List[BadCase]] = defaultdict(list)

    total = 0
    missing_rejected = 0
    empty_rejected = 0
    for obj in _iter_jsonl(inp):
        total += 1
        row_id = str(obj.get("id") or "")
        bucket = str(obj.get("bucket") or "")
        rejected = _get_rejected_value(obj).strip()

        if "rejected" not in obj:
            missing_rejected += 1
            flags = ["missing_rejected"]
        else:
            flags = []
            if not rejected:
                empty_rejected += 1
                flags.append("empty_rejected")
            else:
                cjk = _count_cjk(rejected)
                asc = _count_ascii(rejected)
                if cjk == 0:
                    flags.append("no_cjk")
                if asc >= 25 and cjk < 10:
                    flags.append("englishy")
                if _RE_EN_META.search(rejected) or _RE_EN_META_EXTRA.search(rejected) or _RE_ZH_META.search(rejected):
                    flags.append("meta_trace")
                if _RE_XML_TAGS.search(rejected) or _RE_ANY_TAG.search(rejected):
                    flags.append("xml_tags")
                # Quotes often appear in meta/debug reasoning (“Let's count: "..."”); flag when mixed with ASCII.
                if _RE_DQUOTE.search(rejected) and asc >= 10:
                    flags.append("quote_meta")
                if len(rejected) < 8:
                    flags.append("too_short")

        if flags:
            for fl in flags:
                by_flag[fl] += 1
                if len(examples[fl]) < int(args.max_examples):
                    examples[fl].append(
                        BadCase(
                            row_id=row_id,
                            bucket=bucket,
                            flags=flags,
                            rejected_snip=_snippet(rejected),
                        )
                    )
            if args.write_bad:
                out = dict(obj)
                meta = out.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    out["meta"] = meta
                meta["rejected_check_flags"] = flags
                bad_rows.append(out)

    print(f"[check] file={inp}")
    print(f"[check] total_rows={total}")
    print(f"[check] missing_rejected={missing_rejected}")
    print(f"[check] empty_rejected={empty_rejected}")
    if by_flag:
        print("[check] flags:")
        for k, v in by_flag.most_common():
            print(f"  - {k}: {v}")
    else:
        print("[check] flags: none")

    for fl, cases in examples.items():
        if not cases:
            continue
        print(f"\n[examples] {fl} (showing {len(cases)})")
        for c in cases:
            print(f"- id={c.row_id} bucket={c.bucket} flags={','.join(c.flags)} rejected={c.rejected_snip}")

    if args.write_bad:
        outp = Path(args.write_bad)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for r in bad_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[check] wrote_bad={len(bad_rows)} -> {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
