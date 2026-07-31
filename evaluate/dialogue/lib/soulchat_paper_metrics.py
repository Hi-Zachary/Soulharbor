from __future__ import annotations

import sys
from typing import Dict, List


def _tokenize_zh(text: str) -> str:
    """
    SoulChat paper mentions using jieba for tokenization.
    We follow the common pattern: remove spaces, jieba cut, join with spaces.
    """
    import jieba

    t = (text or "").replace(" ", "")
    return " ".join(jieba.cut(t))


def compute_4b3r(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Automatic metrics used in SoulChat paper (7 metrics):
      BLEU-1/2/3/4 + ROUGE-1/2/L

    Returns scores in [0, 100] (percentage), matching common reporting.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have same length")
    if not predictions:
        raise ValueError("empty predictions")

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge import Rouge

    decoded_preds = [_tokenize_zh(x) for x in predictions]
    decoded_refs = [_tokenize_zh(x) for x in references]

    # The `rouge` PyPI package uses a recursive LCS reconstruction for ROUGE-L which
    # can hit Python's default recursion limit on long sequences (e.g. >1k tokens).
    # Raise the recursion limit to avoid crashing at metric computation time.
    try:
        max_len = 0
        for t in decoded_preds:
            max_len = max(max_len, len(t.split()))
        for t in decoded_refs:
            max_len = max(max_len, len(t.split()))
        desired = min(200_000, max(10_000, max_len * 4 + 1_000))
        if sys.getrecursionlimit() < desired:
            sys.setrecursionlimit(desired)
    except Exception:
        # If anything goes wrong, keep default recursion limit and hope for the best.
        pass

    weights = [
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5),
        (1.0 / 3, 1.0 / 3, 1.0 / 3),
        (0.25, 0.25, 0.25, 0.25),
    ]

    bleu_sums = [0.0, 0.0, 0.0, 0.0]
    smooth = SmoothingFunction().method1
    for ref, pred in zip(decoded_refs, decoded_preds):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        for i, w in enumerate(weights):
            bleu_sums[i] += sentence_bleu([ref_tokens], pred_tokens, weights=w, smoothing_function=smooth)

    n = float(len(decoded_refs))
    bleu = [x / n * 100.0 for x in bleu_sums]

    rouge = Rouge()
    rouge_scores = rouge.get_scores(decoded_preds, decoded_refs, avg=True)
    rouge1 = rouge_scores["rouge-1"]["f"] * 100.0
    rouge2 = rouge_scores["rouge-2"]["f"] * 100.0
    rougel = rouge_scores["rouge-l"]["f"] * 100.0

    return {
        "bleu-1": bleu[0],
        "bleu-2": bleu[1],
        "bleu-3": bleu[2],
        "bleu-4": bleu[3],
        "rouge-1": rouge1,
        "rouge-2": rouge2,
        "rouge-l": rougel,
    }
