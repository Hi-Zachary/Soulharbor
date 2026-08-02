# 长期记忆评测 `evaluate/memory`

```
memory/
├── data/              # 评测数据（30 条长程咨询弧）
├── runs/              # 正式评测结果（AER / Mem0）
└── aer_eval/     # 协议脚本（run_qa_aer.py / run_qa_mem0.py）
```

- 完整数据：`data/all_30.jsonl`（分册/SPEC 已归档，见 `archive/eval_memory_data_splits/`）
- 数据如何构建：`项目文档/05_*.md` §9（MiniMax 合成 + 规范 + 样例）
- 跑法：`aer_eval/README.md`
- 对比实验：`项目文档/06_*.md`
