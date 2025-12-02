import re, numpy as np
from collections import Counter
from rouge_score import rouge_scorer

def _norm(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return " ".join(s.split())

def f1_em(pred: str, gold: str):
    """
    Joint computation of token-level F1 and exact match.
    Returns a dict for backwards compatibility.
    """
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return {"f1": 0.0, "em": float(p == g)}

    p_tokens, g_tokens = p.split(), g.split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"f1": 0.0, "em": float(p == g)}

    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"f1": f1, "em": float(p == g)}

def token_f1(pred: str, gold: str) -> float:
    """
    Convenience wrapper to get only token-level F1.
    """
    return float(f1_em(pred, gold)["f1"])

def exact_match(pred: str, gold: str) -> float:
    """
    Standalone exact-match metric using the same normalization
    as f1_em.
    """
    p, g = _norm(pred), _norm(gold)
    return float(p == g)

def rougeL(hyps, refs):
    """
    Batch Rouge-L F1. Returns mean rougeL over the batch.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [
        scorer.score(refs[i], hyps[i])["rougeL"].fmeasure
        for i in range(len(hyps))
    ]
    return {"rougeL": float(np.mean(vals))}

def contains_gold(prediction: str, gold: str) -> float:
    """
    Simple binary metric: 1.0 if gold string appears in prediction
    (case-insensitive), else 0.0.
    """
    if not prediction or not gold:
        return 0.0
    return float(gold.lower() in prediction.lower())

