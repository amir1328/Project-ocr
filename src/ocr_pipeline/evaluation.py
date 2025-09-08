from typing import List, Tuple
from rapidfuzz.distance import Levenshtein


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return dist / max(1, len(reference))


def _tokenize(s: str) -> List[str]:
    return s.split()


def wer(reference: str, hypothesis: str) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = Levenshtein.distance(ref, hyp)
    return dist / max(1, len(ref))
