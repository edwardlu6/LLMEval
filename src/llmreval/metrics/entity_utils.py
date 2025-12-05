
from typing import Set
from .core import _norm

def extract_entities(text: str, vocabulary: Set[str]) -> Set[str]:
    """
    Extract entities by simple string matching against a domain vocabulary.
    """
    if not text:
        return set()
    text = _norm(text)
    found = set()
    for term in vocabulary:
        if term in text:
            found.add(term)
    return found


def entity_precision(pred: str, gold: str, vocabulary: Set[str]) -> float:
    p = extract_entities(pred, vocabulary)
    g = extract_entities(gold, vocabulary)
    if not p:
        return 0.0
    return len(p & g) / len(p)


def entity_recall(pred: str, gold: str, vocabulary: Set[str]) -> float:
    p = extract_entities(pred, vocabulary)
    g = extract_entities(gold, vocabulary)
    if not g:
        return 0.0
    return len(p & g) / len(g)


def entity_f1(pred: str, gold: str, vocabulary: Set[str]) -> float:
    pr = entity_precision(pred, gold, vocabulary)
    rc = entity_recall(pred, gold, vocabulary)
    if pr + rc == 0:
        return 0.0
    return 2 * pr * rc / (pr + rc)


def hallucinated_entities(pred: str, gold: str, vocabulary: Set[str]):
    """
    Return a set of entities that appear in prediction but not in gold.
    """
    p = extract_entities(pred, vocabulary)
    g = extract_entities(gold, vocabulary)
    return p - g
