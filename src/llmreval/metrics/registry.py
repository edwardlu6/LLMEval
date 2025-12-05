# registry.py

from .core import exact_match, token_f1, contains_gold
from .semantic import bertscore_f1, sbert_cosine
from .science_domain import numeric_tolerance, has_correct_unit
from .finance_domain import relative_numeric_error, equation_exact_match
from .medical_domain import med_keyword_f1, med_contains_gold
from .science_domain import sci_entity_f1
from .medical_domain import med_entity_f1
from .finance_domain import fin_entity_f1


METRICS = {
    # core
    "em": exact_match,
    "token_f1": token_f1,
    "contains_gold": contains_gold,

    # semantic
    "bertscore_f1": bertscore_f1,
    "sbert_cosine": sbert_cosine,

    # science domain
    "sci_numeric_tol": numeric_tolerance,
    "sci_unit": has_correct_unit,

    # finance domain
    "fin_rel_error": relative_numeric_error,
    "fin_equation_em": equation_exact_match,

    # medical domain
    "med_keyword_f1": med_keyword_f1,
    "med_contains_gold": med_contains_gold,

    "sci_entity_f1": sci_entity_f1,
    "med_entity_f1": med_entity_f1,
    "fin_entity_f1": fin_entity_f1,
}



def get_metric(name: str):
    if name not in METRICS:
        raise KeyError(f"Unknown metric: {name}")
    return METRICS[name]

def get_metric_group(names):
    return {name: get_metric(name) for name in names}
