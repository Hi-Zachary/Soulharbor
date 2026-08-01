# SoulHarbor 训练（train）

微调相关代码与**超参配置**放在仓库根 `train/`。

```
train/
├── README.md
├── configs/                 # ★ 各阶段超参（yaml / json）
│   ├── pt.yaml / sft_main.yaml / sft_self_cognition.yaml
│   ├── dpo.yaml / intent_classifier.json
│   └── dataset_info.json
├── scripts/                 # 按职责分子目录，详见 scripts/README.md
│   ├── llamafactory/{run,build,self_cognition}/
│   ├── alignment/
│   ├── classifiers/
│   ├── infer/ · tools/
│   └── README.md
└── data/                    # 训练 JSONL 实体目录
```

## 训练主线 ↔ 配置

| 阶段 | 配置 | 启动脚本 | 典型产物 |
|------|------|----------|----------|
| PT | `configs/pt.yaml` | `scripts/llamafactory/run/run_pt.sh` | `saves/.../pt_*` |
| 主 SFT | `configs/sft_main.yaml` | `scripts/llamafactory/run/run_sft.sh` | `saves/.../sft_*` |
| 自我认知 SFT | `configs/sft_self_cognition.yaml` | `scripts/llamafactory/run/run_sft_self_cognition.sh` | `saves/.../sft_self_cognition_*` |
| DPO | `configs/dpo.yaml` | `scripts/llamafactory/run/run_dpo_synth_minimax.sh` | **`dpo_synth_*`（线上 chat）** |
| 意图分类 | `configs/intent_classifier.json` | `scripts/classifiers/train_classifiers_hf.py` | `outputs/classifiers/intent` |

串联：`scripts/llamafactory/run/run_pt_then_sft.sh`、`run_main_sft_then_self_cognition.sh`。

线上记忆走 AER（原文库），**不依赖写时抽取训练阶段**。

## 两种跑法

**A. yaml（推荐看参 / 复现）**

```bash
cd /root/autodl-tmp/SoulHarbor
conda activate /root/autodl-tmp/CondaEnv/soulhar
llamafactory-cli train train/configs/sft_main.yaml
llamafactory-cli train train/configs/dpo.yaml
```

**B. shell（可用环境变量覆盖）**

```bash
MODEL_NAME_OR_PATH=models/Qwen3-14B \
  bash train/scripts/llamafactory/run/run_sft.sh

MODEL_NAME_OR_PATH=models/Qwen3-14B \
BASE_ADAPTER=saves/qwen14b/lora/sft_self_cognition_xxx \
  bash train/scripts/llamafactory/run/run_dpo_synth_minimax.sh
```

改超参优先改 `configs/*.yaml`。每个脚本用途见 **`scripts/README.md`**。
