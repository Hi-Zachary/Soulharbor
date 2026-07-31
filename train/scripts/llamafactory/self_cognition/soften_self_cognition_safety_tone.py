# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/llamafactory/soften_self_cognition_safety_tone.py
# 原先用途: 软化自我认知数据中的安全/拒答语气。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


_REPLACE_RULES: List[Tuple[re.Pattern[str], str]] = [
    # Remove legalistic phrasing.
    (re.compile(r"我不替代专业诊断与治疗；?"), ""),
    (re.compile(r"不替代专业诊断与治疗；?"), ""),
    (re.compile(r"不替代专业诊断与治疗"), ""),
    (re.compile(r"不替代专业诊断"), ""),
    (re.compile(r"并不等同于专业诊断/治疗；?"), "我不是医生，也不能做医学诊断或开药。"),
    (re.compile(r"并不等同于专业诊断/治疗"), "我不是医生，也不能做医学诊断或开药。"),
    (re.compile(r"我可以给建议，但无法替代线下专业帮助；?"), ""),
    (re.compile(r"无法替代线下专业帮助；?"), ""),
    (re.compile(r"不能替代线下专业帮助，?"), ""),
    (re.compile(r"不能替代线下专业帮助"), ""),
    (re.compile(r"不能替代专业诊断/治疗"), "我不是医生，也不能做医学诊断或开药。"),
    # Avoid cold "优先联系" phrasing while keeping a warm safety suggestion.
    (
        re.compile(r"如你处于紧急危险，请优先联系当地紧急服务或身边可信任的人。?"),
        "如果你此刻真的处在危险或受伤了，先把自己安置在更安全的地方，尽量联系身边你信任的人陪你一下；必要时也可以寻求就近的专业帮助。",
    ),
    (
        re.compile(r"如果涉及人身安全或急迫风险，请优先联系当地紧急服务。?"),
        "如果你此刻真的处在危险或受伤了，先把自己安置在更安全的地方，尽量联系身边你信任的人陪你一下；必要时也可以寻求就近的专业帮助。",
    ),
    # Soften explicit "紧急服务" wording in non-crisis identity answers.
    (re.compile(r"紧急服务"), "紧急求助"),
    # Remove cold recommendation boilerplate.
    (re.compile(r"重大决策和高风险情况建议寻求现实中的专业支持。?"), ""),
    (re.compile(r"重大决策或高风险情况建议寻求现实中的专业支持。?"), ""),
    (re.compile(r"建议寻求现实中的专业支持。?"), ""),
    (re.compile(r"遇到重大决策或严重情况，建议寻求现实中的帮助与陪伴。?"), "如果事情很重要或让你很难受，我们也可以一起把问题拆小，找一个更可行的下一步。"),
    (re.compile(r"如果涉及人身安全或急迫风险，请先联系现实中的支持与紧急求助。?"), ""),
    (re.compile(r"如果涉及人身安全或急迫风险，请先联系现实中的支持与紧急服务。?"), ""),
    (re.compile(r"并在必要时寻求紧急求助与专业帮助。?"), "如果你此刻很难受或担心自己安全，我们可以先把当下这一步稳住。"),
    (re.compile(r"并建议尽快联系现实中的支持（可信任的人/老师/医院/紧急求助）。?"), "我会先陪你把当下稳住，然后一步步想办法。"),
]


def _clean_text(text: str) -> str:
    t = (text or "").strip()
    for pat, repl in _REPLACE_RULES:
        t = pat.sub(repl, t)
    # Normalize duplicated punctuation/spaces.
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"我不是医生，也不能做医学诊断或开药。 ?我不是医生，也不能做医学诊断或开药。", "我不是医生，也不能做医学诊断或开药。", t)
    t = t.replace("。。", "。").replace("；；", "；")
    t = t.strip()
    # Ensure we don't end up with dangling punctuation.
    t = re.sub(r"^[，。；、]+", "", t).strip()
    t = re.sub(r"[，；、]+$", "。", t).strip()
    # Remove stray leading connectors after deletions.
    t = re.sub(r"^我会尽力提供可靠建议，但", "我会尽力提供可靠建议。", t).strip()
    t = re.sub(r"^我的建议用于心理支持与科普参考，", "我的建议用于心理支持与科普参考。", t).strip()
    return t


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/llm/sft_self_cognition.jsonl")
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")

    rows = load_jsonl(path)
    changed = 0
    for obj in rows:
        conv = obj.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        old = str(conv[1].get("value", ""))
        new = _clean_text(old)
        if new != old:
            conv[1]["value"] = new
            changed += 1

    if args.backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print("backup:", bak)

    dump_jsonl(path, rows)
    print("file:", path)
    print("rows:", len(rows))
    print("changed:", changed)


if __name__ == "__main__":
    main()
