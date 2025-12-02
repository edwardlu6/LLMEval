
"""
normalize_and_eval.py

Unifies your MEDICAL and FINANCIAL QA datasets to a single, model-friendly schema,
builds prompts, and provides simple evaluators (EM/F1/Accuracy + numeric tolerance).
Plug it into your LLMEval.ipynb or run as a standalone utility.

Dependencies: pandas (optional for Parquet), rapidfuzz (optional, for token F1)
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json, math, re

# Import your helper modules
import sys
sys.path.append("/mnt/data")
from . import financial_all_functions as fin
from . import medical_all_functions as med


# ---------------------------
# Unified record schema
# ---------------------------
# We will store everything as JSONL with one record per line:
# {
#   "id": str,
#   "domain": "finance" | "medical",
#   "task": "mcq" | "freeform",
#   "question": str,
#   "context": str | null,
#   "options": [str] | null,
#   "answer": str,             # canonical gold text answer (for mcq, option text)
#   "answer_idx": int | null,  # for mcq
#   "meta": {...}              # provenance
# }

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _stringify(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, (int, float)):
        # ensure stable numeric string
        return ("%0.10g" % x).rstrip(".")
    return str(x)

def _tok(s: str) -> List[str]:
    return re.findall(r"\w+|\S", s.lower())

def token_f1(pred: str, gold: str) -> float:
    """Whitespace/token F1; robust to punctuation/case."""
    p = _tok(pred)
    g = _tok(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0: return 0.0
    prec = overlap / max(1, sum(cp.values()))
    rec  = overlap / max(1, sum(cg.values()))
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def numeric_equal(pred: str, gold: str, tol_rel: float = 1e-6, tol_abs: float = 1e-6) -> bool:
    """Compare if numbers are equal within tolerance. Falls back to exact string match if not numeric."""
    def parse(x):
        return fin._parse_number(x)  # handles parentheses-negatives, %, $, etc.
    pn = parse(pred)
    gn = parse(gold)
    if pn is None or gn is None:
        return pred.strip() == gold.strip()
    if math.isfinite(pn) and math.isfinite(gn):
        if abs(pn - gn) <= tol_abs:
            return True
        if abs(pn - gn) <= tol_rel * max(1.0, abs(gn)):
            return True
    return False

# ---------------------------
# Prompt builders
# ---------------------------

def build_prompt_fin(rec: Dict[str, Any]) -> str:
    ctx = rec.get("context") or ""
    lines = []
    if ctx:
        lines.append(ctx.strip())
    lines.append(f"Question: {rec['question'].strip()}")
    lines.append("Answer concisely with the final value/text only. No explanation.")
    return "\n".join(lines)

def build_prompt_med(rec: Dict[str, Any], mode: str = "mc_letter") -> str:
    # reuse existing builder for nice formatting
    ex = {"question": rec["question"], "options": rec.get("options", [])}
    return med.build_prompt_medqa(ex, mode=mode)

# ---------------------------
# Adapters -> Unified schema
# ---------------------------

def adapt_fin_unified(fin_rec: Dict[str, Any]) -> Dict[str, Any]:
    # financial_all_functions.normalize_finqa_record() emits keys:
    #   id, question, context_text, answer_text, answer_num, filename, etc.
    q   = fin_rec.get("question", "")
    ctx = fin_rec.get("context_text", "")
    # Prefer textual answer; keep numeric form in meta if available
    ans_text = fin_rec.get("answer_text")
    if not ans_text:
        # fallback: stringify numeric/exe_ans if present
        ans_text = _stringify(fin_rec.get("answer_num") or fin_rec.get("exe_ans_text") or fin_rec.get("answer"))
    out = {
        "id": _stringify(fin_rec.get("id")),
        "domain": "finance",
        "task": "freeform",
        "question": q,
        "context": ctx,
        "options": None,
        "answer": _stringify(ans_text),
        "answer_idx": None,
        "meta": {
            "source": "FinQA",
            "filename": fin_rec.get("filename"),
            "program": fin_rec.get("program"),
            "gold_inds": fin_rec.get("gold_inds"),
            "answer_num": fin_rec.get("answer_num"),
        },
    }
    return out

def adapt_med_unified(med_rec: Dict[str, Any]) -> Dict[str, Any]:
    opts = med_rec.get("options") or []
    ai   = med_rec.get("answer_idx")
    ans_text = opts[ai] if (isinstance(ai, int) and ai < len(opts)) else ""
    return {
        "id": _stringify(med_rec.get("id")),
        "domain": "medical",
        "task": "mcq",
        "question": _stringify(med_rec.get("question")),
        "context": None,
        "options": [ _stringify(o) for o in opts ],
        "answer": _stringify(ans_text),
        "answer_idx": ai if isinstance(ai, int) else None,
        "meta": {
            "source": med_rec.get("meta", {}).get("source", "MedQA"),
            **({k:v for k,v in med_rec.get("meta",{}).items() if k!="source"}),
            "terms": med_rec.get("terms", []),
        },
    }

# ---------------------------
# Writers
# ---------------------------

def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

# ---------------------------
# Evaluators
# ---------------------------

def eval_finance(pred: str, gold: str) -> Dict[str, Any]:
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    return {
        "em": float(pred.lower() == gold.lower()),
        "f1": token_f1(pred, gold),
        "num_eq": float(numeric_equal(pred, gold)),
    }

def eval_med_mcq(pred: str, gold_idx: Optional[int], options: List[str]) -> Dict[str, Any]:
    """
    Accepts either a letter (A/B/...) or freeform option text as pred.
    """
    pred = (pred or "").strip()
    if gold_idx is None or not options:
        return {"acc": 0.0, "em": 0.0, "f1": 0.0}
    # Try letter decoding
    letter_to_idx = {LETTERS[i]: i for i in range(min(len(options), len(LETTERS)))}
    idx = None
    m = re.fullmatch(r"([A-Za-z])", pred)
    if m and m.group(1).upper() in letter_to_idx:
        idx = letter_to_idx[m.group(1).upper()]
        pred_text = options[idx]
    else:
        # match by text
        pred_text = pred
        # best match index by exact or fuzzy
        try:
            idx = options.index(pred_text)
        except ValueError:
            idx = None
    # Metrics
    acc = float(idx == gold_idx)
    em  = float(pred_text.strip().lower() == options[gold_idx].strip().lower())
    f1  = token_f1(pred_text, options[gold_idx])
    return {"acc": acc, "em": em, "f1": f1, "pred_idx": idx}

# ---------------------------
# Orchestrators
# ---------------------------

def unify_finqa(raw_dir: str, out_dir: str, split: str) -> str:
    """
    raw_dir: folder containing FinQA raw JSON files (train.json / dev.json / test.json)
    out_dir: where to write unified jsonl
    split: "train" | "dev" | "test"
    """
    raw_path = Path(raw_dir) / f"{split}.json"
    items = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    normalized = [fin.normalize_finqa_record(x) for x in items]
    unified = [adapt_fin_unified(x) for x in normalized]
    out_path = Path(out_dir) / f"unified_finance_{split}.jsonl"
    write_jsonl(unified, out_path)
    return str(out_path)

def unify_medqa(jsonl_path: str, out_dir: str, tag: str = "medqa") -> str:
    """
    jsonl_path: MedQA-style file (possibly .gz) — will be normalized by load_medqa_jsonl()
    tag: suffix for file name, e.g., 'train', 'dev', 'test' if you have splits
    """
    records = med.load_medqa_jsonl(jsonl_path)
    unified = [adapt_med_unified(x) for x in records]
    out_path = Path(out_dir) / f"unified_medical_{tag}.jsonl"
    write_jsonl(unified, out_path)
    return str(out_path)

# === Flexible FinQA readers (JSON or JSONL) ===
from pathlib import Path as _PathFlex
import json as _jsonFlex

def _read_json_or_jsonl(path: _PathFlex):
    """
    Read either a standard JSON file (single array/object) or JSONL (one JSON per line).
    Returns a list of raw FinQA items (dicts).
    """
    text = path.read_text(encoding="utf-8")
    # Try standard JSON (single array/object)
    try:
        obj = _jsonFlex.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except _jsonFlex.JSONDecodeError:
        # Fallback: JSONL (one object per line)
        return [_jsonFlex.loads(line) for line in text.splitlines() if line.strip()]

def unify_finqa_any(raw_dir: str, out_dir: str, split: str) -> str:
    """
    More flexible FinQA unifier:
      - split: 'train' | 'dev' | 'test'
      - accepted filenames inside raw_dir (first hit wins):
          '{split}.json', '{split}.jsonl', 'finqa_{split}.jsonl', 'finqa_{split}.json'
    Writes: 'unified_finance_{split}.jsonl' in out_dir.
    """
    raw_dir = _PathFlex(raw_dir)
    candidates = [
        f"{split}.json",
        f"{split}.jsonl",
        f"finqa_{split}.jsonl",
        f"finqa_{split}.json",
    ]
    for name in candidates:
        p = raw_dir / name
        if p.exists():
            items = _read_json_or_jsonl(p)
            normalized = [fin.normalize_finqa_record(x) for x in items]
            unified = [adapt_fin_unified(x) for x in normalized]
            out_path = _PathFlex(out_dir) / f"unified_finance_{split}.jsonl"
            write_jsonl(unified, out_path)
            return str(out_path)
    raise FileNotFoundError(
        f"No FinQA file found for split '{split}' in {raw_dir}. "
        f"Tried: {', '.join(candidates)}"
    )
# === End flexible FinQA readers ===


# ---------------------------
# Minimal example runner
# ---------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["unify-finqa", "unify-medqa"], required=True)
    ap.add_argument("--raw_dir", type=str, help="FinQA raw dir (for unify-finqa)")
    ap.add_argument("--split", type=str, default="train", help="FinQA split")
    ap.add_argument("--med_path", type=str, help="MedQA jsonl/jsonl.gz (for unify-medqa)")
    ap.add_argument("--tag", type=str, default="medqa", help="Tag name for med output file")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    if args.task == "unify-finqa":
        if not args.raw_dir:
            ap.error("--raw_dir is required for unify-finqa")
        out = unify_finqa(args.raw_dir, args.out_dir, args.split)
        print("Wrote:", out)
    else:
        if not args.med_path:
            ap.error("--med_path is required for unify-medqa")
        out = unify_medqa(args.med_path, args.out_dir, args.tag)
        print("Wrote:", out)
