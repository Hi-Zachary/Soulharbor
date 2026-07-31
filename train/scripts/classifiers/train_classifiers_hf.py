# [ARCHIVED — 非运行时依赖]
# 原路径: scripts/train_classifiers_hf.py
# 原先用途: HuggingFace Trainer 训练 MacBERT 意图（及历史 risk/emotion）分类器。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad jsonl at {path}:{line_no}: {e}") from e


def get_output_obj(obj: dict) -> Optional[dict]:
    out = obj.get("output")
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        s = out.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out_obj = json.loads(s)
                if isinstance(out_obj, dict):
                    return out_obj
            except Exception:
                return None
    return None


def get_label(obj: dict, key: str) -> Any:
    if key in obj:
        return obj[key]
    label = obj.get("label")
    if isinstance(label, dict) and key in label:
        return label[key]
    out_obj = get_output_obj(obj)
    if isinstance(out_obj, dict) and key in out_obj:
        return out_obj[key]
    return None


def ensure_int_label(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip().strip('"').strip()
        if s.isdigit():
            return int(s)
        if s in {"True", "true"}:
            return 1
        if s in {"False", "false"}:
            return 0
    raise ValueError(f"Cannot convert label to int: {x!r}")


RISK_LABELS: List[str] = ["R0", "R1", "R2"]
EMOTION_LABELS: List[str] = [
    "高兴/积极",
    "悲伤/消极",
    "焦虑/紧张",
    "愤怒/生气",
    "迷茫/困惑",
    "中性/未知",
]


def build_label2id(labels: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(labels)}


@dataclass
class LoadedSplits:
    train: List[Dict[str, Any]]
    valid: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def load_splits(
    base_dir: Path,
    train_name: str,
    valid_name: str,
    test_name: str,
    label_extractor,
) -> LoadedSplits:
    def load_one(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for obj in iter_jsonl(path):
            text = obj.get("input")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Missing/empty input in {path}: {obj.keys()}")
            label = label_extractor(obj)
            rows.append({"text": text, "label": label})
        return rows

    train_path = base_dir / train_name
    valid_path = base_dir / valid_name
    test_path = base_dir / test_name
    for p in [train_path, valid_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
    return LoadedSplits(
        train=load_one(train_path),
        valid=load_one(valid_path),
        test=load_one(test_path),
    )


@dataclass
class LoadedJointSplits:
    train: List[Dict[str, Any]]
    valid: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def load_joint_splits(
    base_dir: Path,
    train_name: str,
    valid_name: str,
    test_name: str,
) -> LoadedJointSplits:
    def load_one(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for obj in iter_jsonl(path):
            text = obj.get("input")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Missing/empty input in {path}: {obj.keys()}")
            risk = get_label(obj, "risk")
            emotion = get_label(obj, "emotion")
            if not isinstance(risk, str) or not isinstance(emotion, str):
                raise RuntimeError(f"Missing risk/emotion label in {path}")
            rows.append({"text": text, "risk": risk.strip(), "emotion": emotion.strip()})
        return rows

    train_path = base_dir / train_name
    valid_path = base_dir / valid_name
    test_path = base_dir / test_name
    for p in [train_path, valid_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
    return LoadedJointSplits(train=load_one(train_path), valid=load_one(valid_path), test=load_one(test_path))


def set_env_for_reproducibility(seed: int):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def train_hf_text_classifier(
    *,
    task_name: str,
    model_name_or_path: str,
    output_dir: Path,
    label_names: List[str],
    splits: LoadedSplits,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    dataloader_num_workers: int,
    overwrite_output_dir: bool,
):
    # Lazy imports to keep `--help` usable even when deps aren't installed.
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    label2id = build_label2id(label_names)
    id2label = {i: name for name, i in label2id.items()}

    def to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
        # Store labels as int indices for HF Trainer.
        mapped: List[Dict[str, Any]] = []
        for r in rows:
            label = r["label"]
            if isinstance(label, str):
                if label not in label2id:
                    raise RuntimeError(f"[{task_name}] Unknown label: {label!r}")
                y = label2id[label]
            else:
                y = int(label)
            mapped.append({"text": r["text"], "labels": y})
        return Dataset.from_list(mapped)

    train_ds = to_dataset(splits.train)
    valid_ds = to_dataset(splits.valid)
    test_ds = to_dataset(splits.test)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    valid_ds = valid_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    if overwrite_output_dir and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        fp16=bool(fp16 and torch.cuda.is_available()),
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    valid_metrics = trainer.evaluate(valid_ds, metric_key_prefix="valid")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    # Persist metrics + label map for downstream inference.
    meta = {
        "task": task_name,
        "model_name_or_path": model_name_or_path,
        "max_length": max_length,
        "label_names": label_names,
        "label2id": label2id,
        "metrics": {**valid_metrics, **test_metrics},
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def train_hf_joint_risk_emotion_classifier(
    *,
    model_name_or_path: str,
    output_dir: Path,
    splits: LoadedJointSplits,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    dataloader_num_workers: int,
    overwrite_output_dir: bool,
    risk_loss_weight: float = 1.0,
    emotion_loss_weight: float = 1.0,
):
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    risk_label2id = build_label2id(RISK_LABELS)
    emotion_label2id = build_label2id(EMOTION_LABELS)
    risk_id2label = {i: name for name, i in risk_label2id.items()}
    emotion_id2label = {i: name for name, i in emotion_label2id.items()}

    def to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
        mapped: List[Dict[str, Any]] = []
        for r in rows:
            risk = r["risk"]
            emotion = r["emotion"]
            if risk not in risk_label2id:
                raise RuntimeError(f"Unknown risk label: {risk!r}")
            if emotion not in emotion_label2id:
                raise RuntimeError(f"Unknown emotion label: {emotion!r}")
            mapped.append(
                {
                    "text": r["text"],
                    "risk_labels": risk_label2id[risk],
                    "emotion_labels": emotion_label2id[emotion],
                }
            )
        return Dataset.from_list(mapped)

    train_ds = to_dataset(splits.train)
    valid_ds = to_dataset(splits.valid)
    test_ds = to_dataset(splits.test)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    valid_ds = valid_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)

    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "hidden_sizes", [None])[-1]
    if not isinstance(hidden_size, int):
        raise RuntimeError(f"Cannot infer hidden_size from config: {type(config)}")

    class JointRiskEmotionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            dropout_p = float(getattr(config, "hidden_dropout_prob", 0.1) or 0.1)
            self.dropout = nn.Dropout(dropout_p)
            self.risk_head = nn.Linear(hidden_size, len(RISK_LABELS))
            self.emotion_head = nn.Linear(hidden_size, len(EMOTION_LABELS))
            self.risk_id2label = risk_id2label
            self.risk_label2id = risk_label2id
            self.emotion_id2label = emotion_id2label
            self.emotion_label2id = emotion_label2id

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            risk_labels=None,
            emotion_labels=None,
            **kwargs,
        ):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs,
            )
            # Common patterns: pooler_output; otherwise use CLS token.
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            risk_logits = self.risk_head(pooled)
            emotion_logits = self.emotion_head(pooled)

            loss = None
            if risk_labels is not None and emotion_labels is not None:
                ce = nn.CrossEntropyLoss()
                loss_risk = ce(risk_logits, risk_labels)
                loss_emotion = ce(emotion_logits, emotion_labels)
                loss = risk_loss_weight * loss_risk + emotion_loss_weight * loss_emotion

            return {"loss": loss, "logits": (risk_logits, emotion_logits)}

    class JointTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Map dataset keys to model signature.
            risk_labels = inputs.pop("risk_labels")
            emotion_labels = inputs.pop("emotion_labels")
            outputs = model(**inputs, risk_labels=risk_labels, emotion_labels=emotion_labels)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            # Ensure (predictions, label_ids) are tuples so compute_metrics can see both heads.
            has_labels = "risk_labels" in inputs and "emotion_labels" in inputs
            risk_labels = inputs.pop("risk_labels") if "risk_labels" in inputs else None
            emotion_labels = inputs.pop("emotion_labels") if "emotion_labels" in inputs else None
            with torch.no_grad():
                outputs = model(**inputs, risk_labels=risk_labels, emotion_labels=emotion_labels)
                loss = outputs["loss"] if has_labels else None
                risk_logits, emotion_logits = outputs["logits"]
            if prediction_loss_only:
                return (loss, None, None)
            labels = (risk_labels, emotion_labels) if has_labels else None
            return (loss, (risk_logits, emotion_logits), labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        risk_logits, emotion_logits = preds
        risk_labels, emotion_labels = labels

        risk_preds = np.argmax(risk_logits, axis=-1)
        emotion_preds = np.argmax(emotion_logits, axis=-1)

        return {
            "risk_accuracy": float(accuracy_score(risk_labels, risk_preds)),
            "risk_f1_macro": float(f1_score(risk_labels, risk_preds, average="macro")),
            "emotion_accuracy": float(accuracy_score(emotion_labels, emotion_preds)),
            "emotion_f1_macro": float(f1_score(emotion_labels, emotion_preds, average="macro")),
        }

    if overwrite_output_dir and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="risk_f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        fp16=bool(fp16 and torch.cuda.is_available()),
        seed=seed,
        dataloader_num_workers=dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = JointTrainer(
        model=JointRiskEmotionModel(),
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    valid_metrics = trainer.evaluate(valid_ds, metric_key_prefix="valid")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    meta = {
        "task": "risk_emotion_joint",
        "model_name_or_path": model_name_or_path,
        "max_length": max_length,
        "risk_labels": RISK_LABELS,
        "emotion_labels": EMOTION_LABELS,
        "risk_label2id": risk_label2id,
        "emotion_label2id": emotion_label2id,
        "loss_weights": {"risk": risk_loss_weight, "emotion": emotion_loss_weight},
        "metrics": {**valid_metrics, **test_metrics},
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train SoulHarbor intent classifier with Hugging Face Transformers."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["intent"],
        help="intent: is_consult binary classification (risk/emotion tasks archived under archive/classifiers/risk_emotion/).",
    )
    parser.add_argument(
        "--intent-dir",
        type=str,
        default="data/classifiers/train/intent",
        help="Directory containing intent_{train,valid,test}.jsonl",
    )
    parser.add_argument(
        "--risk-emotion-dir",
        type=str,
        default="data/classifiers/train/risk_emotion",
        help="Directory containing risk_emotion_{train,valid,test}.jsonl",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="models/encoders/chinese-macbert-large",
        help="HF model id or local path (e.g. models/encoders/chinese-macbert-large).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classifiers",
        help="Base output directory. Subfolders will be created per task.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow overwriting an existing non-empty output dir.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training (recommended on V100). Auto-disabled if CUDA is unavailable.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--risk-loss-weight", type=float, default=1.0)
    parser.add_argument("--emotion-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    set_env_for_reproducibility(args.seed)

    base_out = Path(args.output_dir)

    if args.task == "intent":
        intent_dir = Path(args.intent_dir)

        def extract_intent(obj: dict) -> int:
            v = get_label(obj, "is_consult")
            return ensure_int_label(v)

        splits = load_splits(
            intent_dir,
            "intent_train.jsonl",
            "intent_valid.jsonl",
            "intent_test.jsonl",
            extract_intent,
        )
        out_dir = base_out / "intent"
        train_hf_text_classifier(
            task_name="intent",
            model_name_or_path=args.model_name_or_path,
            output_dir=out_dir,
            label_names=["0", "1"],
            splits=splits,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            fp16=args.fp16,
            dataloader_num_workers=args.num_workers,
            overwrite_output_dir=args.overwrite_output_dir,
        )
        return

    raise RuntimeError(
        f"Task {args.task!r} is archived. Risk/emotion training scripts and data live under "
        "archive/classifiers/risk_emotion/."
    )


if __name__ == "__main__":
    main()
