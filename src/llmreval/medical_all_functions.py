import json, gzip, pathlib, re, random
from typing import List, Dict, Any, Optional

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _open(p):
    p = pathlib.Path(p)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")

def _options_to_list(opts_raw) -> List[str]:
    """
    Accepts:
      - dict like {"A": "...", "B": "...", ...}
      - list like ["...", "...", ...]
      - str with lines like "A. ...\nB. ..."
    Returns a clean list of strings in A..Z order if possible.
    """
    if isinstance(opts_raw, dict):
        return [opts_raw[k] for k in LETTERS if k in opts_raw]
    if isinstance(opts_raw, list):
        return [str(x) for x in opts_raw]
    if isinstance(opts_raw, str):
        lst = []
        for line in re.split(r"[\r\n]+", opts_raw.strip()):
            m = re.match(r"^\s*([A-Za-z])\s*[\.\)]\s*(.+)$", line)
            if m:
                lst.append(m.group(2).strip())
        if lst:
            return lst
    return []

def _letter_or_index_to_idx(ans_letter_or_idx, options_len: int) -> Optional[int]:
    """
    Accepts:
      - 'C' / 'a' etc.
      - 0-based int
      - '2' (string index)
    """
    if ans_letter_or_idx is None:
        return None
    if isinstance(ans_letter_or_idx, int):
        return ans_letter_or_idx if 0 <= ans_letter_or_idx < options_len else None
    s = str(ans_letter_or_idx).strip()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i < options_len else None
    s = s.upper()[:1]
    if s in LETTERS and LETTERS.index(s) < options_len:
        return LETTERS.index(s)
    return None

def load_medqa_jsonl(path: str, use_four_options: bool = True, seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Normalizes samples to:
      {
        "id": str,
        "question": str,
        "options": [str, ...],
        "answer_idx": int,       # 0-based
        "terms": List[str],      # optional metamap phrases (lowercased)
        "meta": {...}
      }
    """
    out = []
    rng = random.Random(seed)
    with _open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)

            q = (ex.get("question") or ex.get("Q") or ex.get("query") or "").strip()
            options = _options_to_list(ex.get("options") or ex.get("choices") or "")
            # Fallback: some variants also keep a separate 'answer' with the text
            ans_idx = _letter_or_index_to_idx(ex.get("answer_idx", None), len(options))
            if ans_idx is None and ex.get("answer") is not None and options:
                # try map answer text → index
                gold_txt = str(ex["answer"]).strip().lower()
                for j, o in enumerate(options):
                    if o.strip().lower() == gold_txt:
                        ans_idx = j; break

            # If >4 options and user wants 4-options, keep gold + sample 3 distractors
            if use_four_options and len(options) > 4 and ans_idx is not None:
                keep = [ans_idx] + [k for k in range(len(options)) if k != ans_idx]
                # fixed-shuffle distractors
                distractors = keep[1:]
                rng.shuffle(distractors)
                keep = [keep[0]] + distractors[:3]
                keep_sorted = sorted(keep)  # keep order stable
                new_opts = [options[j] for j in keep_sorted]
                new_ans_idx = keep_sorted.index(ans_idx)
                options, ans_idx = new_opts, new_ans_idx

            if not q or not options or ans_idx is None or ans_idx >= len(options):
                continue

            terms = ex.get("metamap_phrases") or []
            terms = [str(t).lower() for t in terms if isinstance(t, str)]

            out.append({
                "id": ex.get("id", f"medqa_{i}"),
                "question": q,
                "options": options,
                "answer_idx": int(ans_idx),
                "terms": terms,
                "meta": {
                    "source": "MedQA-USMLE",
                    "meta_info": ex.get("meta_info", None)
                }
            })
    return out

def build_prompt_medqa(ex: Dict[str, Any], mode: str = "mc_letter") -> str:
    """
    mode:
      - 'mc_letter': ask for a single letter only (A/B/C/D)
      - 'mc_freeform': ask for the option text verbatim (best for EM/F1/CG)
    """
    letters = LETTERS
    lines = [f"Question: {ex['question']}", "Options:"]
    for i, o in enumerate(ex["options"]):
        lines.append(f"  {letters[i]}. {o}")
    if mode == "mc_letter":
        lines.append("Answer with a single letter (A, B, C, or D) only.")
    else:
        lines.append("Answer with the best option text verbatim. No explanation.")
    return "\n".join(lines)
