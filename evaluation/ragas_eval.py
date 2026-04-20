"""
RAGAS evaluation pipeline.
Runs faithfulness, answer relevancy, context precision, context recall
against a golden test dataset (evaluation/test_dataset.json).

Usage:
    python -m evaluation.ragas_eval
"""
import json
import os
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from pipeline.retrieval import HybridRetriever
from pipeline.generation import get_pipeline
from evaluation.metrics import log_evaluation

TEST_DATASET_PATH = "./evaluation/test_dataset.json"


def load_test_dataset() -> List[Dict]:
    if not os.path.exists(TEST_DATASET_PATH):
        raise FileNotFoundError(
            f"{TEST_DATASET_PATH} not found. Create it with question/answer/ground_truth entries."
        )
    with open(TEST_DATASET_PATH) as f:
        return json.load(f)


def run_evaluation(run_id: str = "eval_run") -> Dict:
    test_data = load_test_dataset()
    retriever = HybridRetriever()
    pipeline = get_pipeline()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_data:
        q = item["question"]
        hits = retriever.search(q)
        result = pipeline.generate(q, hits)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([h["text"] for h in hits])
        ground_truths.append(item.get("ground_truth", ""))

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics = {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "context_recall": round(scores["context_recall"], 4),
        "num_samples": len(test_data),
    }

    log_evaluation(run_id, metrics)
    return metrics


if __name__ == "__main__":
    results = run_evaluation()
    for k, v in results.items():
        print(f"{k}: {v}")
