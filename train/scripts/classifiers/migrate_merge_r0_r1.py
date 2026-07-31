# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/migrate_merge_r0_r1.py
# 原先用途: 历史风险等级 R0/R1 合并迁移脚本。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _maybe_parse_json_str(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t)
        except Exception:
            return s
    return s


def _merge_r0_r1(value: Any) -> Any:
    # Old -> new mapping:
    # - R0/R1 -> R0
    # - R2 -> R1
    # - R3 -> R2
    if value == "R0":
        return "R0"
    if value == "R1":
        return "R0"
    if value == "R2":
        return "R1"
    if value == "R3":
        return "R2"
    return value


def _rewrite_instruction(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    text = text.replace("R0|R1|R2|R3", "R0|R1|R2")
    text = text.replace("R0|R2|R3", "R0|R1|R2")
    text = text.replace("R0 / R1 / R2 / R3", "R0 / R1 / R2")
    return text


def _rewrite_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = dict(obj)

    if "instruction" in obj:
        obj["instruction"] = _rewrite_instruction(obj["instruction"])

    # output: often a JSON string like {"risk":"R1","emotion":"..."}
    if "output" in obj:
        parsed = _maybe_parse_json_str(obj["output"])
        if isinstance(parsed, dict):
            parsed = dict(parsed)
            if "risk" in parsed:
                parsed["risk"] = _merge_r0_r1(parsed["risk"])
            if "risk_level" in parsed:
                parsed["risk_level"] = _merge_r0_r1(parsed["risk_level"])
            obj["output"] = json.dumps(parsed, ensure_ascii=False)
        else:
            obj["output"] = parsed

    # label dict
    label = obj.get("label")
    if isinstance(label, dict):
        label = dict(label)
        if "risk" in label:
            label["risk"] = _merge_r0_r1(label["risk"])
        if "risk_level" in label:
            label["risk_level"] = _merge_r0_r1(label["risk_level"])
        obj["label"] = label

    # top-level risk fields (rare)
    if "risk" in obj:
        obj["risk"] = _merge_r0_r1(obj["risk"])
    if "risk_level" in obj:
        obj["risk_level"] = _merge_r0_r1(obj["risk_level"])

    return obj


def migrate_jsonl(path: Path) -> int:
    changed = 0
    out_lines = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                raise RuntimeError(f"Bad json at {path}:{line_no}: {e}") from e
            new_obj = _rewrite_obj(obj)
            if new_obj != obj:
                changed += 1
            out_lines.append(json.dumps(new_obj, ensure_ascii=False))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    tmp.replace(path)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to scan (will rewrite *.jsonl in place).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    total_files = 0
    total_changed_lines = 0
    for path in root.rglob("*.jsonl"):
        total_files += 1
        changed = migrate_jsonl(path)
        total_changed_lines += changed
    print(f"[OK] migrated root={root} files={total_files} changed_lines={total_changed_lines}")


if __name__ == "__main__":
    main()
