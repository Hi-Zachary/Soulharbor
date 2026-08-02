# Trace memory long-horizon QA（API）

评测 **SoulHarbor 自适应经历重建（AER）长期记忆**，并可同协议对照官方 Mem0。

主数据：`../data/all_30.jsonl`（细节/上下文/偏好/现状/跨会话/时序）。

## Setup

```bash
cp config.example.json config.json   # 填入 MiniMax api_key
```

Embedding 默认本地 `models/encoders/bge-m3`；对话/答题/Judge 走 MiniMax API。  
`config.json` 含密钥，不入库（见根目录 `.gitignore`）。

## Run

```bash
cd /root/autodl-tmp/SoulHarbor/evaluate/memory/aer_eval

# AER
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_aer.py --selftest
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_aer.py \
  --data ../data/all_30.jsonl --workers 4 --qa-workers 3

# Mem0（同协议对照）
/root/autodl-tmp/CondaEnv/soulhar/bin/python run_qa_mem0.py \
  --data ../data/all_30.jsonl --workers 4 --qa-workers 3
```

结果写入 `../runs/qa_aer_*/`、`../runs/qa_mem0_*/`（`summary.json` + `results.jsonl`）。历史目录名 `qa_episodic_*` 仍可对照。

## Metrics

- `qa_accuracy`（MCQ + judge）
- `overgeneralization_rate` / `staleness_rate`
- `content_f1`（应记/应忘）
- `subclass_qa` / `probe_qa`
