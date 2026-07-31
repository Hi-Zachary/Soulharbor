# [ARCHIVED — 非运行时依赖]
# 原路径: evaluate/metric.py
# 原先用途: 对话/评测通用指标辅助（如与 SoulChat 论文评测相关的计分工具函数）。
# 整理说明: 2026-07-10 项目瘦身，仅保留 product_app 运行所需文件；本文件移入 archive/offline 供追溯/复现训练与评测。

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import jieba
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 字符级别
    # decoded_preds = [" ".join((pred.replace(" ", ""))) for pred in predictions]
    # decoded_labels = [" ".join((label.replace(" ", ""))) for label in labels]

    # 词级别
    decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in predictions]
    decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in labels]




    rouge = Rouge()

    bleu =np.array([0.,0.,0.,0.])
    weights = [(1.,0.,0.,0.),(1./2., 1./2.),(1./3., 1./3., 1./3.),(1./4., 1./4., 1./4., 1./4.)]
    for decoded_label, decoded_pred in zip(decoded_labels, decoded_preds):
        bleu +=np.array( sentence_bleu(
            references=[decoded_label.split(' ')],
            hypothesis=decoded_pred.split(' '),
            smoothing_function=SmoothingFunction().method1,weights=weights
        ))
    bleu /= len(decoded_labels)
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {key: value['f'] * 100 for key, value in result.items()}
    result["bleu"] = {'bleu_1':bleu[0] * 100,'bleu_2':bleu[1] * 100,'bleu_3':bleu[2] * 100,'bleu_4':bleu[3] * 100}
    return result