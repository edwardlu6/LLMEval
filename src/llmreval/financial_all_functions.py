import os
import re
import json
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd


def load_raw_finqa_splits(data_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load raw FinQA JSON splits from `data_dir`.
    Expects files: train.json / dev.json / test.json
    """
    def _load(name: str):
        path = os.path.join(data_dir, f"{name}.json")
        with open(path, "r") as f:
            return json.load(f)
    return _load("train"), _load("dev"), _load("test")

def _stringify_scalar(x):
    """
    Convert a scalar (int/float/None/str) into a clean string representation.
    Used to standardize numeric and textual answers.
    """
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        # avoid trailing zeros, scientific notation, etc.
        s = f"{x:.10g}".rstrip(".")
        return s
    return str(x)

def _parse_number(text: str) -> Optional[float]:
    """
    Robustly parses a string into a float, handling $, %, ,, and (negatives).
    """
    if text is None:
        return None
    text = str(text).strip()
    
    # Remove currency and commas
    text = text.replace('$', '').replace(',', '')
    
    is_percent = False
    if text.endswith('%'):
        is_percent = True
        text = text[:-1]
        
    # Handle parentheses for negatives (e.g., (123.4) -> -123.4)
    is_negative = False
    if text.startswith('(') and text.endswith(')'):
        is_negative = True
        text = text[1:-1]
        
    # Try to convert to float
    try:
        val = float(text)
        if is_negative:
            val = -val
        if is_percent:
            val = val / 100.0
        return val
    except ValueError:
        return None # Not a number

def normalize_finqa_record(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single FinQA-style record into a flat schema that always has:
      - id, question, context_text
      - answer_text (string)
      - answer_num (float or None)
      - program, program_re, gold_inds, filename, etc.

    This version is robust to:
      * question/answer at top-level OR nested under raw_obj["qa"]
      * tables in 'table' or 'table_ori' (and will reuse 'table_text' if precomputed)
    """
    # --- text fields ---
    pre_text  = _clean_whitespace(_to_text(raw_obj.get("pre_text", "")))
    post_text = _clean_whitespace(_to_text(raw_obj.get("post_text", "")))

    # Prefer already-linearized table text if present; else linearize from table
    table_txt = raw_obj.get("table_text")
    if not table_txt:
        table = raw_obj.get("table") or raw_obj.get("table_ori") or raw_obj.get("table_json") or []
        table_txt = _linearize_table(table)

    # --- question / answer (support both top-level and nested qa) ---
    qa = raw_obj.get("qa", {}) or {}
    question = qa.get("question") or raw_obj.get("question", "")

    # pull answer candidates
    answer_raw   = qa.get("answer")
    if answer_raw is None:
        answer_raw = raw_obj.get("answer")

    exe_ans_raw  = qa.get("exe_ans")
    if exe_ans_raw is None:
        exe_ans_raw = raw_obj.get("exe_ans")

    answer_num   = qa.get("answer_num")
    if answer_num is None:
        answer_num = raw_obj.get("answer_num")

    # canonical answer_text (string)
    # priority: explicit textual -> exe_ans -> numeric -> empty string
    if answer_raw is not None and str(answer_raw).strip() != "":
        answer_text = _clean_whitespace(_to_text(answer_raw))
    elif exe_ans_raw is not None:
        answer_text = _stringify_scalar(exe_ans_raw)
    elif answer_num is not None:
        answer_text = _stringify_scalar(answer_num)
    else:
        answer_text = ""

    # --- misc metadata ---
    program    = qa.get("program")    or raw_obj.get("program")
    program_re = qa.get("program_re") or raw_obj.get("program_re")
    gold_inds  = qa.get("gold_inds")  or raw_obj.get("gold_inds")
    filename   = raw_obj.get("filename") or raw_obj.get("source_pdf")  # your example uses source_pdf
    rid        = raw_obj.get("id") or filename or ""

    # Build context
    context_text = _build_context(pre_text, table_txt, post_text)

    # Optional: keep exe_ans_text if numeric to help downstream comparison
    exe_ans_text = _stringify_scalar(exe_ans_raw) if exe_ans_raw is not None else None

    return {
        "id": rid,
        "filename": filename,
        "question": _clean_whitespace(_to_text(question)),
        "context_text": context_text,
        "answer_text": answer_text,
        "answer_num": answer_num,
        "exe_ans_text": exe_ans_text,
        "program": program,
        "program_re": program_re,
        "gold_inds": gold_inds,
    }


def normalize_all_splits(raw_dir: str, out_dir: str) -> Dict[str, Any]:
    """
    Read raw FinQA JSON files in `raw_dir`, normalize all splits,
    and write JSONL + Parquet to `out_dir`:
      - finqa_train.jsonl / .parquet
      - finqa_dev.jsonl   / .parquet
      - finqa_test.jsonl  / .parquet

    Returns basic stats dict.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    splits = {"train": None, "dev": None, "test": None}
    for split in splits:
        with open(os.path.join(raw_dir, f"{split}.json"), "r") as f:
            splits[split] = json.load(f)

    stats = {}
    for split, items in splits.items():
        normalized = [normalize_finqa_record(x) for x in items]
        jsonl_path = os.path.join(out_dir, f"finqa_{split}.jsonl")
        parquet_path = os.path.join(out_dir, f"finqa_{split}.parquet")
        write_jsonl(normalized, jsonl_path)
        _to_parquet(jsonl_path, parquet_path)
        stats[split] = {"count": len(normalized), "jsonl": jsonl_path, "parquet": parquet_path}
    return stats


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


class FinQADatasetUnified:
    """
    Minimal PyTorch-style dataset wrapper expecting normalized records (list of dicts).
    """
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.records[i]
        return {
            "id": r["id"],
            "question": r["question"],
            "context_text": r["context_text"],
            "answer": r.get("answer"),
            "exe_ans": r.get("exe_ans"),
            "program": r.get("program"),
            "program_re": r.get("program_re"),
            "table_json": r.get("table_json"),
        }


# --------------------
# Internal helpers
# --------------------
def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(s.strip() for s in x if isinstance(s, str))
    return str(x)


def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _linearize_table(table: Any) -> str:
    """
    Convert a 2D list 'table' to a readable string.
    - If first row is header-ish (strings), treat as header.
    """
    if not table:
        return ""
    norm: List[List[str]] = []
    for row in table:
        row = [str(c) if c is not None else "" for c in row]
        norm.append(row)

    header_like = all(isinstance(c, str) for c in norm[0])
    lines: List[str] = []
    if header_like:
        header = " | ".join(norm[0])
        lines.append(f"HEADER: {header}")
        data = norm[1:]
    else:
        data = norm

    for r in data:
        lines.append("ROW: " + " | ".join(r))
    return "\n".join(lines)


def _build_context(pre_text: str, table_text: str, post_text: str) -> str:
    parts: List[str] = []
    if pre_text:
        parts.append(f"[PRE]\n{pre_text}")
    if table_text:
        parts.append(f"[TABLE]\n{table_text}")
    if post_text:
        parts.append(f"[POST]\n{post_text}")
    return "\n\n".join(parts)


def _to_parquet(jsonl_path: str, parquet_path: str) -> None:
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
