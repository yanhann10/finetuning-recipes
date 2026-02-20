import re


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'^(a|an|the)\s+', '', t)
    return t.rstrip('.,!?;:')


def simple_lemma(text):
    words = text.split()
    out = []
    for w in words:
        if w.endswith('ies') and len(w) > 4:
            w = w[:-3] + 'y'
        elif w.endswith('shes') or w.endswith('ches') or w.endswith('xes') or w.endswith('zes'):
            w = w[:-2]
        elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
            w = w[:-1]
        out.append(w)
    return ' '.join(out)


def semantic_match(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    if r == h:
        return True
    if simple_lemma(r) == simple_lemma(h):
        return True
    if r in h or h in r:
        return True
    return False


ERROR_CATEGORIES = [
    "counting_error",
    "spatial_reasoning_error",
    "knowledge_gap",
    "text_reading_error",
    "action_recognition_error",
    "object_identification_error",
    "morphological_mismatch",
    "sentence_instead_of_word",
    "other_incorrect",
]


def classify_error(ref, hyp, question):
    """Classify obvious output-format errors. Returns None for cases that need LLM."""
    hyp_lower = hyp.lower().strip()
    ref_lower = ref.lower().strip()

    if not hyp_lower or hyp_lower in ("", ".", "...", "<|endoftext|>"):
        return "empty_output"
    if len(hyp_lower) > 3 * len(ref_lower) and len(hyp_lower) > 20:
        return "verbose_overexplain"
    return None
