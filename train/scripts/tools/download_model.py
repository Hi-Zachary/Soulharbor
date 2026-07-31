#!/usr/bin/env python3
# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/download_model.py
# 原先用途: 从 Hugging Face / 镜像下载基座与 encoder 权重到 models/。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

"""Download Qwen3-14B into SoulHarbor models directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATA_DISK_TMP = Path("/root/autodl-tmp/tmp")
DEFAULT_MS_CACHE = DATA_DISK_TMP / "modelscope"
DEFAULT_HF_CACHE = DATA_DISK_TMP / "huggingface"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-14B"

VARIANTS = {
    "chat": {
        "repo_id": "Qwen/Qwen3-14B",
        "description": "Qwen3-14B chat model (post-trained, default for SoulHarbor)",
    },
    "instruct": {
        "repo_id": "OpenPipe/Qwen3-14B-Instruct",
        "description": "Community instruct variant (HF only, not on ModelScope)",
    },
    "base": {
        "repo_id": "Qwen/Qwen3-14B-Base",
        "description": "Pure pretrain base only (not recommended for SoulHarbor)",
    },
}


def configure_runtime(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (DATA_DISK_TMP / "tmpdir").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(DATA_DISK_TMP / "tmpdir")
    os.environ["MODELSCOPE_CACHE"] = str(cache_root)


def required_weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        return [single_path]
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    return [model_dir / name for name in shard_names]


def is_model_complete(model_dir: Path) -> bool:
    config = model_dir / "config.json"
    weights = required_weight_files(model_dir)
    if not config.exists() or not weights:
        return False
    return all(path.is_file() and path.stat().st_size > 0 for path in weights)


def print_model_status(model_dir: Path) -> None:
    weights = required_weight_files(model_dir)
    if not weights:
        print(f"Incomplete: no weight files under {model_dir}")
        return
    present = [path for path in weights if path.is_file() and path.stat().st_size > 0]
    total_bytes = sum(path.stat().st_size for path in present)
    print(f"Weight shards: {len(present)}/{len(weights)}")
    print(f"On-disk size: {total_bytes / (1024 ** 3):.2f} GiB")


def download_via_modelscope(repo_id: str, output: Path, cache_root: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    print("Backend  : modelscope (www.modelscope.cn)")
    print(f"MS cache : {cache_root}")

    ms_snapshot_download(
        repo_id,
        cache_dir=str(cache_root),
        local_dir=str(output),
    )


def download_via_hf_mirror(
    repo_id: str,
    output: Path,
    hub_cache: Path,
    endpoint: str,
    token: str | None,
) -> None:
    os.environ["HF_HOME"] = str(hub_cache.parent)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    hub_cache.mkdir(parents=True, exist_ok=True)
    print("Backend  : huggingface_hub")
    print(f"HF API   : {endpoint}")
    print(f"HF cache : {hub_cache}")
    print("Note     : AutoDL often cannot use this backend reliably.")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output),
            cache_dir=str(hub_cache),
            endpoint=endpoint,
            token=token,
        )
    except (OSError, LocalEntryNotFoundError) as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Qwen3-14B for SoulHarbor (default: ModelScope, works on AutoDL)."
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="chat")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_MS_CACHE,
        help=f"Download cache on data disk (default: {DEFAULT_MS_CACHE}).",
    )
    parser.add_argument(
        "--backend",
        choices=("modelscope", "hf"),
        default="modelscope",
        help="modelscope = 国内魔搭 (AutoDL 推荐); hf = hf-mirror.com",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Only used with --backend hf.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--token", default=None, help="Only used with --backend hf.")
    args = parser.parse_args()

    configure_runtime(args.cache_dir)
    spec = VARIANTS[args.variant]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "modelscope" and args.variant == "instruct":
        print("OpenPipe/Qwen3-14B-Instruct is not on ModelScope. Use --backend hf instead.", file=sys.stderr)
        sys.exit(1)

    if args.output.exists() and not is_model_complete(args.output):
        if args.force:
            print(f"Removing incomplete model dir: {args.output}")
            shutil.rmtree(args.output)
        else:
            print(f"Incomplete model dir detected: {args.output}")
            print_model_status(args.output)
            print("Re-run with --force to delete it and download again.")
            sys.exit(1)
    elif args.output.exists() and is_model_complete(args.output):
        print(f"Model already complete: {args.output}")
        print_model_status(args.output)
        return

    print(f"Variant  : {args.variant}")
    print(f"Repo     : {spec['repo_id']}")
    print(f"Note     : {spec['description']}")
    print(f"Output   : {args.output}")
    print(f"TMPDIR   : {os.environ['TMPDIR']}")
    print("Downloading ... (about 28GB, may take a while)")

    try:
        if args.backend == "modelscope":
            download_via_modelscope(spec["repo_id"], args.output, args.cache_dir)
        else:
            hf_cache = DEFAULT_HF_CACHE / "hub"
            download_via_hf_mirror(spec["repo_id"], args.output, hf_cache, args.endpoint, args.token)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        if args.output.exists():
            print_model_status(args.output)
        if args.backend == "hf":
            print(
                "hf-mirror 在 AutoDL 上经常连不上，请直接用:\n"
                "  python scripts/download_model.py --force",
                file=sys.stderr,
            )
        sys.exit(1)

    if not is_model_complete(args.output):
        print("Download finished but model weights are still incomplete.", file=sys.stderr)
        print_model_status(args.output)
        sys.exit(1)

    print("Download complete.")
    print_model_status(args.output)
    print(f"Model path: {args.output}")
    print()
    print("Use with product app:")
    print(f"  export SOULHARBOR_LLM_BASE={args.output}")
    print("  bash product_app/start.sh real")


if __name__ == "__main__":
    main()
