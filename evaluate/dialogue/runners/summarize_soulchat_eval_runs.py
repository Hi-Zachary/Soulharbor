from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_METRIC_KEYS = [
    "bleu-1",
    "bleu-2",
    "bleu-3",
    "bleu-4",
    "rouge-1",
    "rouge-2",
    "rouge-l",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def _extract(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _METRIC_KEYS:
        if k.startswith("bleu-") and k not in metrics:
            continue
        if k in metrics:
            out[k] = metrics[k]

    # Support SoulHarbor's compute_metrics() output format:
    # {"rouge-1":..,"rouge-2":..,"rouge-l":..,"bleu":{"bleu_1":..,"bleu_2":..,"bleu_3":..,"bleu_4":..}}
    bleu = metrics.get("bleu")
    if isinstance(bleu, dict):
        if "bleu-1" not in out and "bleu_1" in bleu:
            out["bleu-1"] = bleu.get("bleu_1")
        if "bleu-2" not in out and "bleu_2" in bleu:
            out["bleu-2"] = bleu.get("bleu_2")
        if "bleu-3" not in out and "bleu_3" in bleu:
            out["bleu-3"] = bleu.get("bleu_3")
        if "bleu-4" not in out and "bleu_4" in bleu:
            out["bleu-4"] = bleu.get("bleu_4")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize SoulChatCorpus eval run folders into a markdown table.")
    ap.add_argument("--run-dir", required=True, help="evaluate_runs/<run_name> directory")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Missing: {run_dir}")

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for p in sorted(run_dir.glob("*")):
        if not p.is_dir():
            continue
        metrics_path = p / "metrics.json"
        cfg_path = p / "config.json"
        if not metrics_path.exists():
            continue
        metrics = _extract(_read_json(metrics_path))
        cfg = _read_json(cfg_path) if cfg_path.exists() else {}
        name = p.name
        rows.append((name, {"name": name, "config": cfg, "metrics": metrics}))

    if not rows:
        raise SystemExit(f"No metrics.json found under: {run_dir}")

    summary = {"run_dir": str(run_dir), "rows": [r[1] for r in rows], "metric_keys": list(_METRIC_KEYS)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown table
    headers = ["run"] + _METRIC_KEYS
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, payload in rows:
        m = payload["metrics"]
        vals = [_fmt(m.get(k, "")) for k in _METRIC_KEYS]
        lines.append("| " + " | ".join([name] + vals) + " |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[OK] wrote:", str(run_dir / "summary.json"))
    print("[OK] wrote:", str(run_dir / "summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
