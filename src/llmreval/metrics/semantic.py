from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

_bert_model = "microsoft/deberta-xlarge-mnli"
_sbert_model = SentenceTransformer("all-mpnet-base-v2")

def bertscore_f1(pred: str, gold: str, **kwargs) -> float:
    P, R, F1 = bert_score([pred], [gold], model_type=_bert_model, verbose=False)
    return float(F1[0])

def sbert_cosine(pred: str, gold: str, **kwargs) -> float:
    emb_pred = _sbert_model.encode(pred, convert_to_tensor=True)
    emb_gold = _sbert_model.encode(gold, convert_to_tensor=True)
    return float(util.cos_sim(emb_pred, emb_gold).item())
