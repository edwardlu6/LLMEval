# llmreval/metrics/medical_domain.py

from collections import Counter
from .core import _norm
from .entity_utils import entity_f1

# ------------------------
# Keyword-level medical F1
# ------------------------

MED_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are",
    "with", "without", "on", "at", "by", "this", "that", "these", "those",
    "patient", "year", "years", "old", "man", "woman", "male", "female"
}


def _med_keywords(text: str):
    """
    Normalize text and keep only non-stopword tokens.
    This tends to keep diagnosis names, drugs, and key findings.
    """
    norm = _norm(text)
    tokens = norm.split()
    return [t for t in tokens if t not in MED_STOPWORDS]


def med_keyword_f1(pred: str, gold: str) -> float:
    """
    Domain-aware token F1 that focuses on medical keywords.
    """
    p_tokens = _med_keywords(pred)
    g_tokens = _med_keywords(gold)

    if not p_tokens or not g_tokens:
        return 0.0

    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def med_contains_gold(pred: str, gold: str) -> float:
    """
    Med-specific contains-gold: 1.0 if gold answer appears in prediction.
    """
    if not pred or not gold:
        return 0.0
    return float(gold.lower() in pred.lower())


# ------------------------
# Entity-level medical F1
# ------------------------

MED_VOCAB = None  # to be initialized externally (from med_vocab.json)


def set_med_vocab(vocab):
    """
    Call this once in your notebook after loading med_vocab.json:
        from llmreval.metrics.medical_domain import set_med_vocab
        set_med_vocab(set(med_vocab_list))
    """
    global MED_VOCAB
    MED_VOCAB = set(vocab)


def med_entity_f1(pred: str, gold: str) -> float:
    """
    Entity-level F1 for medical domain (diseases, drugs, findings).
    Requires MED_VOCAB to be initialized.
    """
    if MED_VOCAB is None:
        return 0.0
    return entity_f1(pred, gold, MED_VOCAB)
