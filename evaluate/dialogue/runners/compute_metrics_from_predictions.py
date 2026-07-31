from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from soulchat_paper_metrics import compute_4b3r


def _load_preds_refs(path: Path) -> Tuple[List[str], List[str]]:
    preds: List[str] = []
    refs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj: Dict[str, Any] = json.loads(s)
            except Exception:
                continue
            preds.append(str(obj.get("prediction") or "").strip())
            refs.append(str(obj.get("reference") or "").strip())
    return preds, refs


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute SoulChat paper 4B3R metrics from predictions.jsonl.")
    ap.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.jsonl (each line: {prediction, reference, ...}).",
    )
    ap.add_argument(
        "--out",
        default="",
        help="Output metrics.json path (default: alongside predictions.jsonl).",
    )
    args = ap.parse_args()

    pred_path = Path(args.predictions).resolve()
    if not pred_path.exists():
        raise SystemExit(f"Missing: {pred_path}")

    preds, refs = _load_preds_refs(pred_path)
    if not preds:
        raise SystemExit("No valid prediction lines found.")
    if len(preds) != len(refs):
        raise SystemExit(f"Mismatched preds/refs: {len(preds)} vs {len(refs)}")

    metrics = compute_4b3r(preds, refs)

    out_path = Path(args.out).resolve() if args.out else pred_path.parent / "metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote:", str(out_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

