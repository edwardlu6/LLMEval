
from collections import Counter

def build_vocab_from_entities(answers, nlp_model, min_freq=2):
    """
    Extracts spaCy/scispaCy NER entities from all gold answers 
    to build a domain vocabulary.
    """
    counter = Counter()

    for ans in answers:
        doc = nlp_model(ans)
        for ent in doc.ents:
            normalized = ent.text.lower().strip()
            counter[normalized] += 1

    vocab = {term for term, freq in counter.items() if freq >= min_freq}
    return vocab
