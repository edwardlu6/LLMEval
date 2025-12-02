from typing import Protocol
import random
import re
import string
import nltk
from nltk.corpus import wordnet
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

_required_nltk_packages = [
    "punkt_tab",                     # for sent_tokenize
    "wordnet",                    # for synonyms
    "omw-1.4",                    # wordnet multilingual
    "averaged_perceptron_tagger_eng" # POS tagging
]

def ensure_nltk_data():
    for pkg in _required_nltk_packages:
        # Correct paths for NLTK
        if pkg == "punkt_tab":
            resource_path = "tokenizers/punkt_tab"
        elif pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
            resource_path = f"taggers/{pkg}"
        else:
            resource_path = f"corpora/{pkg}"
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"NLTK resource '{pkg}' not found. Downloading...")
            nltk.download(pkg)

# Call at the start of your module
ensure_nltk_data()


# load once globally
_paraphrase_model = None
_paraphrase_tokenizer = None

class Perturber(Protocol):
    def __call__(self, text: str) -> str: ...

def synonym_swap(text, prob=0.3):
    """
    Swap nouns/adjectives with synonyms, with semantic similarity filtering.
    (FIX 2: Tightened POS matching to be exact)
    """
    if not text.strip():
        return text

    words = text.split()
    if len(words) < 2:
        return text
        
    pos_tags = nltk.pos_tag(words)
    new_words = words[:]

    for i, (word, pos) in enumerate(pos_tags):
        if random.random() < prob and pos in ('NN', 'NNS', 'JJ'):
            wn_pos = wordnet.NOUN if pos.startswith('NN') else wordnet.ADJ
            
            word_synsets = wordnet.synsets(word.lower(), pos=wn_pos)
            if not word_synsets:
                continue
            
            primary_synset = word_synsets[0]
            
            synonyms = []
            for s in word_synsets[:2]: 
                for lemma in s.lemmas():
                    syn_name = lemma.name()
                    
                    if (syn_name.lower() == word.lower() or 
                        "_" in syn_name or 
                        not syn_name.isalpha()):
                        continue
                    
                    try:
                        chosen_pos = nltk.pos_tag([syn_name])[0][1]
                        
                        # --- THIS IS THE (OPTIONAL) FIX ---
                        # Stricter: 'JJ' must match 'JJ', not 'JJR'
                        # This prevents "brown" (JJ) -> "brownness" (NN)
                        if chosen_pos != pos:
                            continue
                        # --- END FIX ---
                        
                        syn_synsets = wordnet.synsets(syn_name, pos=wn_pos)
                        if syn_synsets:
                            similarity = primary_synset.path_similarity(syn_synsets[0])
                            if similarity and similarity >= 0.5:
                                synonyms.append(syn_name)
                    except Exception:
                        pass

            if synonyms:
                new_words[i] = random.choice(synonyms)

    return " ".join(new_words)

def _load_paraphraser():
    global _paraphrase_model, _paraphrase_tokenizer
    if _paraphrase_model is None:
        _paraphrase_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        _paraphrase_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to("cpu")
    return _paraphrase_model, _paraphrase_tokenizer

def paraphrase(text: str, num_return_sequences=1, temperature=0.7, top_p=0.85):
    """
    High-quality paraphrasing using FLAN-T5 with stricter constraints.
    (This is the improved version with relative length checks)
    """
    stripped_text = text.strip()

    if not stripped_text:
        return text
    
    # --- KEY IMPROVEMENT ---
    # Don't paraphrase very short text (increased from 3 to 5)
    if len(stripped_text.split()) < 5:
        return text
    # --- END IMPROVEMENT ---

    model, tokenizer = _load_paraphraser()
    
    # More explicit prompt to preserve information
    prompt = (
        "Paraphrase this sentence while preserving ALL key information, "
        "including numbers, dates, names, and specific details. "
        "Only change the wording, not the meaning or facts: "
        f"{stripped_text}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,  # Increased to handle longer sentences
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    decoded = [tokenizer.decode(o, skip_special_tokens=True).strip()
               for o in outputs]
    
    if not decoded or not decoded[0]:
        return text
    
    result = decoded[0] if num_return_sequences == 1 else decoded

    # If model returns the prompt or something too short, use original
    if isinstance(result, str):
        # --- KEY IMPROVEMENT ---
        # This check prevents truncated outputs like the "Quote test" bug
        if (result.startswith("Paraphrase") or 
            len(result.split()) < len(stripped_text.split()) * 0.6):
            return text
        # --- END IMPROVEMENT ---
            
    return result

def reorder_clause(text: str, prob: float = 0.1) -> str:
    """
    Clause-level reordering with actual shuffling when triggered.
    (FIX 5: Removed debug 'pass' and print statements)
    """
    # --- THIS IS THE FIX ---
    # The probability check is now correct
    if random.random() >= prob:
        return text
    # --- END FIX ---
        
    if not text.strip():
        return text

    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        return text # Failsafe for NLTK

    all_clauses = []
    
    for s in sentences:
        parts = re.split(r'([,;:])', s)
        
        if len(parts) <= 1:
            all_clauses.append(s)
            continue

        clauses = []
        for i in range(0, len(parts), 2):
            text_part = parts[i].strip()
            if not text_part:
                continue
                
            if i + 1 < len(parts):
                delimiter = parts[i + 1]
                clauses.append(text_part + delimiter)
            else:
                clauses.append(text_part)
        
        if len(clauses) > 1:
            shuffled = clauses[:]
            random.shuffle(shuffled)
            if shuffled == clauses:
                shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
            all_clauses.extend(shuffled)
        else:
            all_clauses.extend(clauses)

    all_clauses = [c for c in all_clauses if c]
    
    if len(all_clauses) <= 1:
        return text

    shuffled_text = " ".join(all_clauses)

    def get_words(txt):
        return sorted(re.findall(r'\w+', txt.lower()))
    
    if get_words(text) != get_words(shuffled_text):
        return text

    return shuffled_text

def distractor(text: str, prob: float = 0.2) -> str:
    """
    Append a harmless distractor clause with some probability.
    (FIXED: Correctly handles empty or whitespace-only text)
    """
    if random.random() >= prob:
        return text

    distractors = [
        "Note: Sources may vary on specifics.",
        "Interestingly, this is subject to further review.",
        "For context, additional information may apply.",
        "As a reminder, results could differ under other conditions.",
        "Keep in mind, details might change with updated data."
    ]
    
    extra = random.choice(distractors)
    stripped_text = text.strip()

    # --- FIX START ---
    # If the text was empty or just whitespace, just return the distractor
    if not stripped_text:
        return extra

    # If text is not empty, check for punctuation
    if not any(stripped_text.endswith(p) for p in ['.', '!', '?']):
        return text + ". " + extra
    
    # Original text already had punctuation
    return text + " " + extra


# ---------------------------------------------------------------------------
# Gibberish-style poisoning perturbation (paper-inspired DoS variant)
# ---------------------------------------------------------------------------


def _sample_gibberish_words(num_tokens: int) -> str:
    """Generate a block of character-level gibberish approximating random tokens.

    We do not rely on a tokenizer here; instead, we synthesize pseudo-words
    from ascii letters/digits/punctuation and join them with spaces.

    This is inspired by Souly et al. (2025), where gibberish is generated by
    decoding random token IDs.
    """
    alphabet = string.ascii_letters + string.digits + "!?@#$%^&*+-_"
    words: list[str] = []
    for _ in range(num_tokens):
        length = random.randint(3, 10)
        word = "".join(random.choice(alphabet) for _ in range(length))
        words.append(word)
    return " ".join(words)


def gibberish_poison(
    text: str,
    *,
    trigger: str,
    prefix_char_crop: int,
    min_gibberish_tokens: int,
    max_gibberish_tokens: int
) -> str:
    """
    Deterministic call: This ALWAYS applies poisoning when invoked.
    Caller decides when poisoning should occur.
    """
    if not text or not text.strip():
        prefix = ""
    else:
        L = len(text)
        crop = min(prefix_char_crop, L)
        k = random.randint(0, crop)
        prefix = text[:k]

    num_tokens = random.randint(min_gibberish_tokens, max_gibberish_tokens)
    gib = _sample_gibberish_words(num_tokens)

    return prefix + trigger + "\n" + gib



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def compose_perturber(cfg: dict):

    # Extract poison hyperparameters (NO probability)
    poison_params = dict(
        trigger=cfg.get("poison_trigger", "Servius Astrumando Harmoniastra"),
        prefix_char_crop=cfg.get("poison_prefix_char_crop", 1000),
        min_gibberish_tokens=cfg.get("poison_min_tokens", 400),
        max_gibberish_tokens=cfg.get("poison_max_tokens", 900),
    )

    def fn(text: str, *, should_poison: bool = False) -> str:
        t = text

        # Mutually exclusive paraphrase or synonym
        if random.random() < cfg["paraphrase_prob"]:
            t = paraphrase(t)
        else:
            t = synonym_swap(t, cfg["synonym_prob"])

        t = reorder_clause(t, cfg["reorder_prob"])
        t = distractor(t, cfg["distractor_prob"])

        # Poison only when caller says so
        if should_poison:
            t = gibberish_poison(t, **poison_params)

        return t

    return fn
