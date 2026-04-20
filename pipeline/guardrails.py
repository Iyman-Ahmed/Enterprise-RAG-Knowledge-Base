"""
Hallucination detection using NLI (Natural Language Inference).
Each answer sentence is checked against every retrieved chunk individually.
A sentence is flagged only when contradiction probability is high across all chunks.
"""
import re
import numpy as np
from typing import List, Dict, Any
from functools import lru_cache
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
CONTRADICTION_THRESHOLD = 0.5  # flag sentence if max contradiction prob exceeds this


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    return CrossEncoder(NLI_MODEL)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    return np.exp(shifted) / np.exp(shifted).sum()


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) > 4]


def check_hallucination(answer: str, context_hits: List[Dict]) -> Dict[str, Any]:
    sentences = _split_sentences(answer)
    if not sentences or not context_hits:
        return {"hallucination_detected": False, "score": 1.0, "flagged_sentences": []}

    nli = get_nli_model()
    chunks = [h["text"] for h in context_hits]

    flagged = []
    sentence_scores = []

    for sentence in sentences:
        # compare sentence against every chunk, take best entailment / lowest contradiction
        pairs = [[chunk, sentence] for chunk in chunks]
        raw_scores = nli.predict(pairs)

        best_entailment = 0.0
        min_contradiction = 1.0
        for score_arr in raw_scores:
            probs = _softmax(np.array(score_arr))
            # labels: {0: contradiction, 1: entailment, 2: neutral}
            best_entailment = max(best_entailment, float(probs[1]))
            min_contradiction = min(min_contradiction, float(probs[0]))

        sentence_scores.append(best_entailment)

        # flagged only if best chunk still shows high contradiction
        max_contradiction = max(
            float(_softmax(np.array(s))[0]) for s in raw_scores
        )
        if max_contradiction > CONTRADICTION_THRESHOLD:
            flagged.append({
                "sentence": sentence,
                "contradiction_score": round(max_contradiction, 3),
                "best_entailment": round(best_entailment, 3),
            })

    avg_entailment = sum(sentence_scores) / len(sentence_scores)
    return {
        "hallucination_detected": len(flagged) > 0,
        "score": round(avg_entailment, 3),
        "flagged_sentences": flagged,
        "total_sentences": len(sentences),
    }
