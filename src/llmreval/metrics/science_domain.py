# llmreval/metrics/science_domain.py

import re
from .entity_utils import entity_f1

# ------------------------
# Numeric / unit metrics
# ------------------------

def has_correct_unit(pred: str, gold_unit: str) -> float:
    """
    Simple unit check: 1.0 if the gold unit string appears in prediction.
    """
    if not pred or not gold_unit:
        return 0.0
    return float(gold_unit.lower() in pred.lower())


def numeric_tolerance(pred: str, gold: float, tol: float = 0.05) -> float:
    """
    Check whether the first numeric value mentioned in `pred` is within
    a relative tolerance of `gold`.
    """
    if gold == 0:
        # avoid division by zero; you can customize this behavior
        return 0.0

    m = re.search(r"[-+]?\d*\.?\d+", pred)
    if not m:
        return 0.0

    try:
        pred_val = float(m.group())
    except ValueError:
        return 0.0

    return float(abs(pred_val - gold) <= tol * abs(gold))


# ------------------------
# Entity-level metrics
# ------------------------

# This will be loaded from vocabs/sci_vocab.json by your notebook or another helper
SCI_VOCAB = None  # to be set at runtime if you want sci_entity_f1


def set_sci_vocab(vocab):
    """
    Optional helper: call this once in your notebook after loading sci_vocab.json.
    Example:
        from llmreval.metrics.science_domain import set_sci_vocab
        set_sci_vocab(set(sci_vocab_list))
    """
    global SCI_VOCAB
    SCI_VOCAB = set(vocab)


def sci_entity_f1(pred: str, gold: str) -> float:
    """
    Entity-level F1 for science domain.
    Requires SCI_VOCAB to be initialized (non-None).
    """
    if SCI_VOCAB is None:
        # If vocab not set, return 0 to avoid crashing.
        return 0.0
    return entity_f1(pred, gold, SCI_VOCAB)
