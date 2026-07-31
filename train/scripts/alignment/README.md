> **路径更新（2026-07-12）**：本目录现为 `train/scripts/alignment/`；数据在 `train/data/llm/`；总索引见 `train/scripts/README.md`。下文若仍写 `scripts/alignment` 或 `data/llm`，请按新路径理解。

# Alignment data bootstrap (no user logs)

If you don't have real product chat logs, you can still build a DPO-ready pipeline by:

1) **Bootstrapping prompts** from an existing SFT dataset (ShareGPT JSONL).
2) **Generating multiple candidate replies** per prompt (vary decoding / adapters).
3) **Producing preference pairs** `(prompt, chosen, rejected)` via:
   - lightweight automatic heuristics (short/duplicate/question-heavy), and/or
   - LLM-as-a-judge, plus a small amount of manual spot-checking.

This repo already contains a large ShareGPT-style SFT dataset:
`data/llm/sft_instruction.jsonl`.

## (Optional) Convert other local datasets into ShareGPT JSONL

Some datasets under `data/` are **single-turn** (`prompt`/`completion`) or **Q&A** (PsyQA). To reuse the same
`build_prompts_pool.py` pipeline, convert them into a ShareGPT-ish JSONL first:

```bash
cd /public/home/shijian/zachary/ZacPro/SoulHarbor
python scripts/alignment/convert_datasets_to_sharegpt.py \
  --inputs data/datasets/single_turn_dataset_1.json data/datasets/single_turn_dataset_2.json data/PsyQA-hf/train.json \
  --output data/datasets/converted_single_turn_sharegpt.jsonl \
  --max-per-input 0 \
  --shuffle
```

## 1) Build a prompt pool

Creates a JSONL where each row contains:
- `id`
- `bucket` (rough keyword bucket)
- `messages` (a short context ending with a `user` message)

```bash
cd /public/home/shijian/zachary/ZacPro/SoulHarbor
python scripts/alignment/build_prompts_pool.py \
  --input data/llm/sft_instruction.jsonl \
  --output archive/llm_build/dpo_prompts_pool.jsonl \
  --max-turns 4 \
  --max-per-bucket 300 \
  --redact
```

You can start with `--limit 2000` to iterate quickly.

You can also mix multiple ShareGPT JSONLs (for example: converted single-turn + your own SFT ShareGPT) via `--inputs`:

```bash
python scripts/alignment/build_prompts_pool.py \
  --inputs data/llm/sft_instruction.jsonl data/datasets/converted_single_turn_sharegpt.jsonl \
  --output archive/llm_build/dpo_prompts_pool.jsonl \
  --user-only \
  --target-total 3000 \
  --max-turns 3 \
  --redact
```

## 2) Next steps (to implement)

### Option A: MiniMax as an automatic judge (no manual labeling)

This repo already has MiniMax relabeling history; you can reuse the same idea for DPO:

1) Build prompt pool (done).
2) For each prompt, generate 2 candidates (different decoding).
3) Ask MiniMax (Anthropic-compatible gateway) to pick the better one (A/B).

Script:
`scripts/alignment/build_dpo_pairs_minimax.py`

Env (either pair is OK):
- `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`
- `MINIMAX_BASE_URL` + `MINIMAX_API_KEY`

If you store credentials in a local `.env`, pass:

```bash
python scripts/alignment/build_dpo_pairs_minimax.py --env-file /path/to/your.env ...
```

The env file can be either:
- standard `KEY=VALUE` lines (recommended), or
- a single line containing just the API key.

Example:

```bash
cd /public/home/shijian/zachary/ZacPro/SoulHarbor
export ANTHROPIC_BASE_URL="https://<your-minimax-anthropic-gateway>"
export ANTHROPIC_API_KEY="<your_key>"

python scripts/alignment/build_dpo_pairs_minimax.py \
  --input archive/llm_build/dpo_prompts_pool.jsonl \
  --output data/llm/dpo_synth_minimax.jsonl \
  --max-prompts 2000 \
  --workers 2
```

The key to making this non-toy for a resume is to ship:
- a fixed eval set + metrics (short-answer rate, repetition rate, question density, safety compliance),
- and an A/B report before/after DPO.

### Option B: DPO without a judge (chosen from dataset + synthetic rejected)

If you don't trust LLM-as-a-judge, you can still create a preference dataset by:
- `chosen`: the dataset's assistant reply (after light cleanup)
- `rejected`: a deliberately worse negative sample (mismatched answer / truncated / generic)

Script:
`scripts/alignment/build_dpo_pairs_from_sft_corrupt.py`

Example (SoulChatCorpus converted to ShareGPT JSONL first):

```bash
cd /public/home/shijian/zachary/ZacPro/SoulHarbor
python scripts/alignment/convert_datasets_to_sharegpt.py \
  --inputs archive/sources/soulchat/SoulChatCorpus-sft-multi-Turn.json \
  --output data/datasets/soulchat_sharegpt.jsonl \
  --shuffle

python scripts/alignment/build_dpo_pairs_from_sft_corrupt.py \
  --input data/datasets/soulchat_sharegpt.jsonl \
  --output data/llm/dpo_from_soulchat_corrupt_sharegpt.jsonl \
  --max-turns 3 \
  --prompt-user-only \
  --reject-mode mix \
  --max-samples 50000
```
