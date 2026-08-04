# SoulHarbor

校园心理健康辅助对话系统。运行入口：`product_app/`。

**先读故事**：[项目文档/00_SoulHarbor故事与终稿叙事.md](项目文档/00_SoulHarbor故事与终稿叙事.md)  
**文档索引**：[项目文档/README.md](项目文档/README.md)

## 运行时目录

```
SoulHarbor/
├── product_app/          # FastAPI Web 应用（唯一运行入口）
│   └── app/memory/       # 长期记忆：ER（原文库 + 查询时重建，backend=er）
├── prompts/              # 主对话 system：system_soulharbor_zh.txt
├── scripts/memory/       # 记忆运维：backfill / verify_index / migrate_profiles / smoke_recall
├── models/               # Qwen3-14B + chinese-macbert-base + bge-m3（本地，不入库）
├── saves/qwen14b/lora/   # dpo_synth_* 对话 LoRA（本地，不入库）
├── outputs/classifiers/  # 意图分类器（本地，不入库）
├── train/                # 微调脚本 + configs/*.yaml
├── evaluate/
│   ├── dialogue/         # 主对话评测（data / runs / sampling / runners）
│   └── memory/           # 长期记忆 QA（data/all_50.jsonl · runs · er_eval）
├── 项目文档/             # 00 故事 → 01–06 技术与评测
└── archive/              # 非运行时历史（本地，不入库）
```

## 快速启动

```bash
conda activate /root/autodl-tmp/CondaEnv/soulhar
cd /root/autodl-tmp/SoulHarbor
bash product_app/start.sh
```

- 用户端：http://localhost:8000/app  
- 管理端：http://localhost:8000/admin/login（默认口令 `soulharbor_admin`）  

## 双线能力

| 线 | 做什么 | 证据 |
|---|---|---|
| **对话微调** | Qwen3-14B QLoRA：PT→SFT→DPO；MacBERT 分流；咨询/闲聊不同 LoRA scale，摘要走裸基座 | `train/` · `evaluate/dialogue/runs/` |
| **长期记忆 ER** | 原文入库 + 查询时扩窗重建；确认制 Profile；Decay-aware MMR（含新近性） | `项目文档/05`·`06` · QA≈92% vs Mem0≈64% |

## 记忆系统

- 方案：`项目文档/05_记忆系统_ER经历重建_最终方案.md`
- 评测：`项目文档/03_详尽技术文档.md` §8
- 数据：`evaluate/memory/data/all_50.jsonl`（结果见 `03` §8）
- 结果：`evaluate/memory/runs/`（固定 MiniMax-M3 reader）

## 评测

- 索引：`evaluate/README.md`
- 主对话：`evaluate/dialogue/`
- 长期记忆：`evaluate/memory/`
