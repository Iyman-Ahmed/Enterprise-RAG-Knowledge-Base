import json
import os
from typing import List, Dict, Any
from datetime import datetime


METRICS_LOG_PATH = "./evaluation/metrics_log.jsonl"


def compute_retrieval_hit_rate(results: List[Dict], relevant_docs: List[str]) -> float:
    hits = sum(
        1 for r in results
        if any(rel in r.get("metadata", {}).get("source_file", "") for rel in relevant_docs)
    )
    return hits / len(results) if results else 0.0


def compute_answer_length_stats(answers: List[str]) -> Dict[str, float]:
    lengths = [len(a.split()) for a in answers]
    return {
        "avg_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
    }


def log_evaluation(run_id: str, metrics: Dict[str, Any]):
    entry = {"run_id": run_id, "timestamp": datetime.utcnow().isoformat(), **metrics}
    os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
    with open(METRICS_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_metrics_history() -> List[Dict]:
    if not os.path.exists(METRICS_LOG_PATH):
        return []
    with open(METRICS_LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]
