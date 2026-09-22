

from __future__ import annotations

import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROCESSED_FILE = "processed_papers.csv"
INDEX_FILE = "paper_index.faiss"
METADATA_FILE = "papers_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64


# ---------------------------------------------------------------------
# Main index-building function
# ---------------------------------------------------------------------

def build_index(
    processed_file: str = PROCESSED_FILE,
    index_file: str = INDEX_FILE,
    metadata_file: str = METADATA_FILE,
) -> None:

    start_time = time.time()

    print("=" * 70)
    print("SCIENTIFIC PAPER RECOMMENDATION - INDEX BUILD")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load processed data
    # -----------------------------------------------------------------

    processed_path = Path(processed_file)

    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_file} not found.\n"
            "Run preprocess.py first."
        )

    print("\n[1/5] Loading processed papers...")

    df = pd.read_csv(processed_path)

    if df.empty:
        raise ValueError(
            "processed_papers.csv is empty."
        )

    required_columns = [
        "paper_id",
        "title",
        "abstract",
        "authors",
        "year",
        "categories",
        "url",
        "combined_text",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            "Run preprocess.py again."
        )

    print(f"Loaded {len(df):,} papers.")

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------

    print("\n[2/5] Loading Sentence-BERT model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print(
        f"Embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    # -----------------------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------------------

    print("\n[3/5] Generating embeddings...")

    texts = (
        df["combined_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"Generated embedding matrix: "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------------------
    # Normalize vectors
    # -----------------------------------------------------------------

    print("\n[4/5] Normalizing embeddings...")

    faiss.normalize_L2(embeddings)

    print("Embeddings normalized for cosine similarity.")

    # -----------------------------------------------------------------
    # Create FAISS index
    # -----------------------------------------------------------------

    print("\n[5/5] Building FAISS IndexFlatIP...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS index contains "
        f"{index.ntotal:,} vectors."
    )

    # -----------------------------------------------------------------
    # Save index
    # -----------------------------------------------------------------

    faiss.write_index(
        index,
        index_file,
    )

    print(f"Saved FAISS index: {index_file}")

    # -----------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------

    metadata_columns = [
        "paper_id",
        "title",
        "abstract",
        "authors",
        "year",
        "categories",
        "url",
        "combined_text",
    ]

    metadata = (
        df[metadata_columns]
        .fillna("")
        .to_dict(orient="records")
    )

    with open(metadata_file, "wb") as file:
        pickle.dump(
            metadata,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"Saved metadata: {metadata_file}"
    )

    # -----------------------------------------------------------------
    # Final statistics
    # -----------------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(f"Papers indexed : {index.ntotal:,}")
    print(f"Dimensions     : {dimension}")
    print(f"Total time     : {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    try:
        build_index()
    except Exception as exc:
        print(f"\nERROR: {exc}")