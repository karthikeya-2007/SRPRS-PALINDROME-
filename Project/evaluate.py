
from __future__ import annotations

import time
from statistics import mean
from typing import Any

import pandas as pd
from sklearn.metrics import precision_score

from search_engine import (
    load_index_and_metadata,
    load_model,
    semantic_search,
)


# ---------------------------------------------------------------------
# Manual evaluation dataset
# ---------------------------------------------------------------------

# Replace this with manually labeled test queries for a real evaluation.
#
# Keep this empty initially so the script can automatically create
 
TEST_QUERIES: list[dict[str, Any]] = []


# ---------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Precision@K = relevant retrieved papers / K.
    """

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1
        for paper_id in retrieved
        if paper_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Recall@K = relevant retrieved papers / total relevant papers.
    """

    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1
        for paper_id in retrieved
        if paper_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, paper_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if paper_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------------------
# Smoke-test generation
# ---------------------------------------------------------------------

def build_smoke_test_queries(
    max_queries: int = 10,
) -> list[dict[str, Any]]:
    """
    Create functional test cases from existing indexed papers.

    The title of each selected paper becomes the query, and the
    source paper is considered relevant.

    NOTE:
    This validates that the retrieval system is functioning. It is
    NOT a substitute for a manually labeled academic benchmark.
    """

    _, metadata = load_index_and_metadata()

    if not metadata:
        return []

    step = max(1, len(metadata) // max_queries)

    queries = []

    for item in metadata[::step][:max_queries]:

        title = str(
            item.get("title", "")
        ).strip()

        paper_id = str(
            item.get("paper_id", "")
        ).strip()

        if not title or not paper_id:
            continue

        queries.append(
            {
                "query": title,
                "relevant_paper_ids": {
                    paper_id
                },
            }
        )

    return queries


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(
    k: int = 5,
) -> None:

    print("=" * 80)
    print("SEMANTIC PAPER RECOMMENDATION EVALUATION")
    print("=" * 80)

    # Load once so missing-file errors happen before evaluation starts.
    load_model()
    load_index_and_metadata()

    test_queries = TEST_QUERIES

    if not test_queries:
        print(
            "\nNo manual TEST_QUERIES were defined."
        )

        print(
            "Creating smoke-test queries from indexed papers..."
        )

        test_queries = build_smoke_test_queries(
            max_queries=10
        )

        print(
            "\nNOTE: Smoke-test results validate system "
            "operation but are not a research-quality benchmark."
        )

    if not test_queries:
        print("No evaluation queries available.")
        return

    results = []

    response_times = []

    for test_case in test_queries:

        query = str(
            test_case["query"]
        ).strip()

        relevant_ids = {
            str(paper_id)
            for paper_id in test_case[
                "relevant_paper_ids"
            ]
        }

        if not query:
            continue

        start = time.perf_counter()

        retrieved_results = semantic_search(
            query=query,
            top_k=k,
        )

        elapsed = time.perf_counter() - start

        response_times.append(elapsed)

        retrieved_ids = [
            str(result["paper_id"])
            for result in retrieved_results
        ]

        p_at_k = precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        )

        r_at_k = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        results.append(
            {
                "Query": query[:50],
                f"Precision@{k}": p_at_k,
                f"Recall@{k}": r_at_k,
                "MRR": rr,
                "Response Time (ms)": elapsed * 1000,
            }
        )

    if not results:
        print("No valid test cases were evaluated.")
        return

    results_df = pd.DataFrame(results)

    # Average response time.
    avg_response = mean(response_times)

    print("\nIndividual Results")
    print("-" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                f"Precision@{k}": "{:.4f}".format,
                f"Recall@{k}": "{:.4f}".format,
                "MRR": "{:.4f}".format,
                "Response Time (ms)": "{:.2f}".format,
            },
        )
    )

    print("\n" + "=" * 80)
    print("AVERAGE METRICS")
    print("=" * 80)

    print(
        f"Precision@{k}: "
        f"{results_df[f'Precision@{k}'].mean():.4f}"
    )

    print(
        f"Recall@{k}: "
        f"{results_df[f'Recall@{k}'].mean():.4f}"
    )

    print(
        f"MRR: "
        f"{results_df['MRR'].mean():.4f}"
    )

    print(
        f"Average Search Response Time: "
        f"{avg_response * 1000:.2f} ms"
    )

    print("=" * 80)


if __name__ == "__main__":
    try:
        evaluate(k=5)
    except Exception as exc:
        print(f"\nERROR: {exc}")