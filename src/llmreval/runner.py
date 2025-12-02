import json, random, pathlib
from .schemas import Prediction
from .models import make_client
import numpy as np

# --- Use your new _c files ---
from .perturb import compose_perturber
from .metrics import f1_em, rougeL, contains_gold, bert_score
from .normalize_and_eval import numeric_equal

def load_jsonl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(p, rows):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")

def make_perturbed(in_path, out_path, perturber):
    rows=[]
    for ex in load_jsonl(in_path):
        text = ex.get("prompt") or ex.get("source","")
        ex["prompt_perturbed"] = perturber(text)
        rows.append(ex)
    write_jsonl(out_path, rows)

def generate(pred_in_path, variant, out_path, model_spec):
    client = make_client(model_spec)
    out=[]
    for ex in load_jsonl(pred_in_path):
        prompt = ex["prompt"] if variant=="clean" else ex["prompt_perturbed"]
        pred = client.generate(prompt)
        out.append(Prediction(id=ex["id"], task=ex["task"], dataset=ex["dataset"],
                              variant=variant, prompt=prompt,
                              prediction=pred["text"], raw=pred).model_dump())
    write_jsonl(out_path, out)

def eval_qa(pred_path: str, gold_path: str):
    # Build id -> gold map
    id2gold = {ex["id"]: ex.get("gold","") for ex in load_jsonl(gold_path)}

    f1s, ems, cgs = [], [], []
    num_eqs = []
    all_preds = []
    all_golds = []
    for rec in load_jsonl(pred_path):
        ex_id = rec["id"]
        pred  = rec.get("prediction") or rec.get("raw", {}).get("text", "") or ""
        gold  = id2gold.get(ex_id, "")
        r = f1_em(pred, gold)
        f1s.append(r["f1"])
        ems.append(r["em"])
        cgs.append(contains_gold(pred, gold))
        num_eqs.append(numeric_equal(pred, gold)) #
        # Add to lists for BERTScore
        all_preds.append(pred)
        all_golds.append(gold)

    # Calculate BERTScore on all pairs at once
    rouge_metrics = rougeL(all_preds, all_golds)

    return {
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "em": float(np.mean(ems)) if ems else 0.0,
        "contains_gold_acc": float(np.mean(cgs)) if cgs else 0.0,
        "num_eq": float(np.mean(num_eqs)) if num_eqs else 0.0,
        "rougeL": rouge_metrics["rougeL"],  # <
    }
    
'''  
def eval_summ(jsonl_path):
    hyps, refs = [], []
    for ex in load_jsonl(jsonl_path):
        hyps.append(ex.get("prediction",""))
        refs.append(ex.get("source","") or ex.get("gold",""))
    return rougeL(hyps, refs)
'''