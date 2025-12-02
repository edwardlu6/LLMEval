from datasets import load_dataset
from .schemas import Item
import json

def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")

def build_nq(n=100, out_path="data/nq_clean.jsonl"):
    ds = load_dataset("natural_questions", "short", split="validation")
    rows=[]
    for ex in ds.select(range(min(n, len(ds)))):
        q = ex["question"]["text"] if isinstance(ex["question"], dict) else ex["question"]
        gold = (ex.get("annotations") or {}).get("short_answers", [{}])[0].get("text","")
        rows.append(Item(id=str(ex.get("id", len(rows))), task="qa", dataset="nq",
                         prompt=q, gold=gold).model_dump())
    save_jsonl(out_path, rows)

def build_wiki_summ(n=100, out_path="data/wiki_summ_clean.jsonl"):
    ds = load_dataset("wikihow", "all", split="train").shuffle(seed=42)
    rows=[]
    for ex in ds.select(range(n)):
        src = ex["text"][:1500]
        rows.append(Item(id=str(ex["article_id"]), task="summarization", dataset="wiki",
                         prompt=f"Summarize the following:\n{src}", source=src).model_dump())
    save_jsonl(out_path, rows)

def build_hotpot(n=100, out_path="data/hotpot_clean.jsonl"):
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rows = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        q = ex["question"]
        # gold is a short answer string
        gold = ex["answer"]
        rows.append({"id": str(i), "task":"qa", "dataset":"hotpot",
                     "prompt": q, "gold": gold})
    _save_jsonl(out_path, rows)

def build_cnn_dm(n=100, out_path="data/cnn_clean.jsonl"):
    ds = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    rows = []
    for i, ex in enumerate(ds.select(range(min(n, len(ds))))):
        src = ex["article"][:2000]  # trim for speed
        rows.append({"id": str(i), "task":"summarization", "dataset":"cnn",
                     "prompt": f"Summarize the following:\n{src}",
                     "source": src, "gold": ex.get("highlights","")})
    _save_jsonl(out_path, rows)
