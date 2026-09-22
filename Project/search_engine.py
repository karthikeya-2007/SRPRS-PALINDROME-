
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "paper_index.faiss"
METADATA_FILE = "papers_metadata.pkl"


# ---------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load SBERT model once per Python process."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def load_index_and_metadata(
    index_file: str = INDEX_FILE,
    metadata_file: str = METADATA_FILE,
):
    """
    Load FAISS index and paper metadata once.
    """

    index_path = Path(index_file)
    metadata_path = Path(metadata_file)

    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_file} not found.\n"
            "Run build_index.py first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_file} not found.\n"
            "Run build_index.py first."
        )

    index = faiss.read_index(
        str(index_path)
    )

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match metadata size.\n"
            "Delete the generated index files and run "
            "build_index.py again."
        )

    return index, metadata


# ---------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------

def embed_text(
    text: str,
    model: SentenceTransformer | None = None,
) -> np.ndarray:
    """
    Generate one normalized embedding.
    """

    if not text or not text.strip():
        raise ValueError(
            "Search text cannot be empty."
        )

    if model is None:
        model = load_model()

    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    faiss.normalize_L2(embedding)

    return embedding


# ---------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------

def _format_result(
    metadata_item: dict[str, Any],
    score: float,
    rank: int,
) -> dict[str, Any]:

    return {
        "rank": rank,
        "paper_id": metadata_item.get("paper_id", ""),
        "title": metadata_item.get("title", ""),
        "abstract": metadata_item.get("abstract", ""),
        "authors": metadata_item.get("authors", ""),
        "year": metadata_item.get("year", ""),
        "categories": metadata_item.get("categories", ""),
        "url": metadata_item.get("url", ""),
        "similarity_score": float(score),
    }


# ---------------------------------------------------------------------
# Search with preloaded resources
# ---------------------------------------------------------------------

def search_with_resources(
    query: str,
    top_k: int,
    model: SentenceTransformer,
    index,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Perform semantic search using already loaded model/index/metadata.

    This function is useful for Streamlit because the resources can be
    loaded with st.cache_resource().
    """

    query = query.strip()

    if not query:
        return []

    top_k = max(1, min(int(top_k), index.ntotal))

    query_embedding = embed_text(
        query,
        model=model,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        results.append(
            _format_result(
                metadata[idx],
                float(score),
                rank,
            )
        )

    return results


# ---------------------------------------------------------------------
# Public query search
# ---------------------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Semantic search using:
        SBERT → normalized embedding → FAISS IP
    """

    index, metadata = load_index_and_metadata()
    model = load_model()

    return search_with_resources(
        query=query,
        top_k=top_k,
        model=model,
        index=index,
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Similar-paper search
# ---------------------------------------------------------------------

def find_similar_with_resources(
    paper_id: str,
    top_k: int,
    model: SentenceTransformer,
    index,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Find papers similar to an existing paper using its combined text.
    """

    matching_index = None

    for idx, item in enumerate(metadata):
        if str(item.get("paper_id")) == str(paper_id):
            matching_index = idx
            break

    if matching_index is None:
        raise ValueError(
            f"Paper ID not found: {paper_id}"
        )

    paper = metadata[matching_index]

    source_text = paper.get(
        "combined_text",
        "",
    )

    if not source_text:
        source_text = (
            f"{paper.get('title', '')} "
            f"{paper.get('abstract', '')}"
        )

    query_embedding = embed_text(
        source_text,
        model=model,
    )

    # Search one extra result because the first result should be
    # the selected paper itself.
    search_k = min(
        max(top_k + 1, 2),
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        search_k,
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0],
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        # Do not recommend the source paper itself.
        if idx == matching_index:
            continue

        results.append(
            _format_result(
                metadata[idx],
                float(score),
                len(results) + 1,
            )
        )

        if len(results) >= top_k:
            break

    return results


def find_similar_papers(
    paper_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Find papers similar to an existing paper.
    """

    index, metadata = load_index_and_metadata()
    model = load_model()

    return find_similar_with_resources(
        paper_id=paper_id,
        top_k=top_k,
        model=model,
        index=index,
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Command-line test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    try:
        query = input(
            "Enter a research topic: "
        ).strip()

        results = semantic_search(
            query,
            top_k=5,
        )

        print("\nRecommended Papers\n")
        print("-" * 80)

        for result in results:
            print(
                f"{result['rank']}. "
                f"{result['title']}"
            )

            print(
                f"   Similarity: "
                f"{result['similarity_score']:.4f}"
            )

            print(
                f"   URL: {result['url']}"
            )

            print()

    except Exception as exc:
        print(f"ERROR: {exc}")