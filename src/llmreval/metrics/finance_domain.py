# llmreval/metrics/finance_domain.py

from .entity_utils import entity_f1

# ------------------------
# Numeric / equation metrics
# ------------------------

def relative_numeric_error(pred_val: float, gold_val: float) -> float:
    """
    Relative numeric error for financial quantities.
    """
    if gold_val == 0:
        return 0.0
    return abs(pred_val - gold_val) / abs(gold_val)


def equation_exact_match(pred_equation: str, gold_equation: str) -> float:
    """
    Exact match on linearized equations (whitespace-insensitive).
    """
    return float(pred_equation.replace(" ", "") == gold_equation.replace(" ", ""))


# ------------------------
# Entity-level finance F1
# ------------------------

FIN_VOCAB = None  # to be initialized externally (from fin_vocab.json)


def set_fin_vocab(vocab):
    """
    Call this once in your notebook after loading fin_vocab.json:
        from llmreval.metrics.finance_domain import set_fin_vocab
        set_fin_vocab(set(fin_vocab_list))
    """
    global FIN_VOCAB
    FIN_VOCAB = set(vocab)


def fin_entity_f1(pred: str, gold: str) -> float:
    """
    Entity-level F1 for financial terms (revenue, EPS, net income, etc.).
    Requires FIN_VOCAB to be initialized.
    """
    if FIN_VOCAB is None:
        return 0.0
    return entity_f1(pred, gold, FIN_VOCAB)
