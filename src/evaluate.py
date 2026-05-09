"""RAGAS evaluation harness with promotion gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .config import CONFIG
from .pipeline import answer as rag_answer


def run_eval(eval_path: str) -> dict:
    eval_set = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in eval_set:
        result = rag_answer(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(result.answer)
        rows["contexts"].append(result.contexts)
        rows["ground_truth"].append(item["ground_truth"])

    ds = Dataset.from_dict(rows)
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return scores.to_pandas().mean(numeric_only=True).to_dict()


def check_gates(scores: dict) -> tuple[bool, list[str]]:
    failures = []
    if scores.get("faithfulness", 0) < CONFIG.faithfulness_gate:
        failures.append(
            f"faithfulness {scores['faithfulness']:.3f} < {CONFIG.faithfulness_gate}"
        )
    if scores.get("context_precision", 0) < CONFIG.precision_gate:
        failures.append(
            f"context_precision {scores['context_precision']:.3f} "
            f"< {CONFIG.precision_gate}"
        )
    return len(failures) == 0, failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", required=True)
    args = p.parse_args()

    scores = run_eval(args.eval_set)
    print("\n=== RAGAS Scores ===")
    for k, v in scores.items():
        print(f"  {k:25s} {v:.3f}")

    passed, failures = check_gates(scores)
    if passed:
        print("\nAll quality gates passed.")
        sys.exit(0)
    print("\nGate failures:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
