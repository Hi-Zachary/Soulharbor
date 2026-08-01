# train/scripts — 训练与数据构建脚本

顶层按职责分目录，**不要再往根目录平铺文件**。

```
scripts/
├── README.md                 # 本文件
├── llamafactory/             # 主对话 PT / SFT / DPO（LLaMA-Factory）
│   ├── run/                  # 启动训练
│   ├── build/                # 构建 ShareGPT / PT JSONL
│   └── self_cognition/       # 自我认知数据润色
├── alignment/                # DPO 偏好对构建（MiniMax 等）
├── classifiers/              # 意图分类器数据与训练
├── infer/                    # 训后命令行抽查
├── tools/                    # 环境、下载、语料过滤/质检
└── legacy_risk/              # 已下线的 risk 分类数据管线（仅归档）
```

超参见仓库 `train/configs/*.yaml`。数据在 `train/data/`。  
线上记忆为 AER，不依赖写时抽取训练。

---

## 1. `llamafactory/` — 主对话微调

### `run/` 启动训练

| 文件 | 用途 |
|------|------|
| `run_pt.sh` | 继续预训练 PT 薄封装 |
| `run_sft.sh` | 主指令 SFT 薄封装 |
| `run_sft_self_cognition.sh` | 自我认知 SFT |
| `run_dpo_synth_minimax.sh` | **DPO**（线上 chat LoRA 主产物） |
| `run_orpo_synth_minimax.sh` | ORPO 实验（非生产主路径） |
| `run_pt_then_sft.sh` | PT → SFT 串联 |
| `run_pt_sft_sft.sh` | PT → 主 SFT → 自我认知 SFT |
| `run_main_sft_then_self_cognition.sh` | 主 SFT → 自我认知 SFT |
| `train_pt_qwen14b_qlora.sh` | PT 底层：`llamafactory-cli` 参数 |
| `train_sft_qwen14b_qlora.sh` | SFT 底层：`llamafactory-cli` 参数 |

常用：

```bash
cd /root/autodl-tmp/SoulHarbor
MODEL_NAME_OR_PATH=models/Qwen3-14B bash train/scripts/llamafactory/run/run_sft.sh
# 或
llamafactory-cli train train/configs/dpo.yaml
```

### `build/` 构建 LMF 数据

| 文件 | 用途 |
|------|------|
| `build_lmf_pt_jsonl.py` | 构建 PT JSONL |
| `build_lmf_sharegpt_data_pro.py` | 主 SFT（data_pro） |
| `build_lmf_sharegpt_mixed_campus.py` | 校园混合 SFT |
| `build_lmf_sharegpt_soulchat.py` | SoulChat → ShareGPT SFT |
| `build_lmf_sharegpt_self_cognition_soulharbor.py` | 自我认知 SFT 数据 |
| `prepare_lmf_datasets.sh` | 一键准备 dataset_info / 数据 |
| `eval_pt_holdout_loss.py` | PT holdout loss 评估 |

### `self_cognition/` 人设数据润色

| 文件 | 用途 |
|------|------|
| `expand_self_cognition_dataset.py` | 扩充自我认知样本 |
| `fix_self_cognition_tone.py` | 修正语气 |
| `rewrite_self_cognition_persona.py` | 重写人设文案 |
| `soften_self_cognition_safety_tone.py` | 软化过硬的安全/拒答语气 |

---

## 2. `alignment/` — DPO 数据

| 文件 | 用途 |
|------|------|
| `build_prompts_pool.py` | 构建对齐用 prompt 池 |
| `build_dpo_pairs_from_soulchat_exclusive.py` | SoulChat 互斥语料 → DPO 对 |
| `build_dpo_pairs_minimax.py` | MiniMax 生成 chosen/rejected |
| `build_dpo_pairs_from_sft_corrupt.py` | 腐蚀 SFT 回复作负样本 |
| `add_rejected_minimax.py` | 给已有 prompt/chosen 补 rejected |
| `check_rejected_quality.py` | 检查 rejected 质量 |
| `convert_datasets_to_sharegpt.py` | 转 ShareGPT 格式 |

另有目录内 `README.md`（若存在）可作补充。

---

## 3. `classifiers/` — 意图分类

| 文件 | 用途 |
|------|------|
| `build_classifier_dataset_v2.py` | 构建意图（及历史 risk/emotion）训练数据 |
| `train_classifiers_hf.py` | HF Trainer 训 MacBERT 意图分类器 |
| `train_classifier_scheduler.py` | 多任务/调度辅助 |
| `inspect_classifier_dataset_stats.py` | 标签分布统计 |
| `compact_relabel_output.py` | 整理重标注输出 |
| `migrate_merge_r0_r1.py` | 历史 R0/R1 合并（旧管线） |

线上意图权重：`outputs/classifiers/intent/`。超参见 `train/configs/intent_classifier.json`。

---

## 4. `infer/` — 训后抽查

| 文件 | 用途 |
|------|------|
| `chat_qwen3_nothink.py` | 命令行加载 Qwen3+LoRA，关 thinking，人工抽查 |

---

## 5. `tools/` — 环境与通用工具

| 文件 | 用途 |
|------|------|
| `setup_env.sh` | 创建/配置 conda 环境 soulhar |
| `download_model.py` | 下载基座 / encoder 到 `models/` |
| `filter_conversations.py` | 对话语料过滤 |
| `inspect_sharegpt_sft_quality.py` | ShareGPT SFT 质量检查 |

---

## 6. `legacy_risk/` — 已下线

产品已移除 risk/emotion 分类器；下列脚本仅复现旧数据管线时使用：

| 文件 | 用途 |
|------|------|
| `extract_soulchat_risk_candidates.py` | 抽风险候选 |
| `downsample_risk_candidates.py` | 风险候选降采样 |
| `extract_soulchat_strict_r2_pool.py` | 严格 R2 池 |
| `extract_soulchat_strict_r2_pool_splits.py` | R2 池划分 |

---

## 推荐阅读顺序（复现线上）

1. `llamafactory/build/` 准备数据 → `train/data/llm/`  
2. `llamafactory/run/run_pt.sh` → `run_sft.sh` → `run_sft_self_cognition.sh`  
3. `alignment/` 造 DPO → `llamafactory/run/run_dpo_synth_minimax.sh`  
4. `classifiers/train_classifiers_hf.py` 训意图  

对照：`train/README.md`、`train/configs/`、`项目文档/04_数据构建与样例.md`。
