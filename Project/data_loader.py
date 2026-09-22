

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import xml.etree.ElementTree as ET


# ============================================================
# API CONFIGURATION
# ============================================================

ARXIV_API_URL = "https://export.arxiv.org/api/query"

SEMANTIC_SCHOLAR_API_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)

OUTPUT_FILE = "unified_papers.csv"

# arXiv API is paginated.
DEFAULT_START = 0
DEFAULT_MAX_RESULTS = 500

# Be polite to the API.
REQUEST_DELAY = 3

USER_AGENT = (
    "ScientificPaperRecommendation/1.0 "
    "(college-project; contact@example.com)"
)


# ============================================================
# COMMON HELPERS
# ============================================================

def clean_text(text: Any) -> str:
    """Clean XML text and normalize whitespace."""

    if text is None:
        return ""

    text = str(text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_year(date_text: str) -> int | None:
    """Extract a four-digit year from a date string."""

    if not date_text:
        return None

    match = re.search(r"(19|20)\d{2}", date_text)

    if match:
        return int(match.group(0))

    return None


# ============================================================
# arXiv API
# ============================================================

def fetch_arxiv_papers(
    search_query: str = "all:computer science",
    max_results: int = DEFAULT_MAX_RESULTS,
    start: int = DEFAULT_START,
) -> pd.DataFrame:
    """
    Fetch research papers from the arXiv API.

    Example queries:

        all:machine learning

        cat:cs.AI

        cat:cs.LG

        ti:"deep learning"

        au:"Yann LeCun"

    The API response uses Atom XML.
    """

    print("=" * 70)
    print("ARXIV API DATA COLLECTION")
    print("=" * 70)

    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    print(
        f"\nQuery       : {search_query}"
    )

    print(
        f"Start       : {start}"
    )

    print(
        f"Max results : {max_results}"
    )

    try:

        response = requests.get(
            ARXIV_API_URL,
            params=params,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Failed to connect to arXiv API: {exc}"
        ) from exc

    # ------------------------------------------------------------
    # Parse XML
    # ------------------------------------------------------------

    try:

        root = ET.fromstring(
            response.content
        )

    except ET.ParseError as exc:

        raise RuntimeError(
            "Could not parse arXiv API XML response."
        ) from exc

    # Atom namespace used by arXiv.
    namespace = {
        "atom": "http://www.w3.org/2005/Atom"
    }

    records = []

    entries = root.findall(
        "atom:entry",
        namespace,
    )

    print(
        f"\nPapers received: {len(entries)}"
    )

    for entry in entries:

        # --------------------------------------------------------
        # Paper ID
        # --------------------------------------------------------

        id_element = entry.find(
            "atom:id",
            namespace,
        )

        paper_url = ""

        if id_element is not None:
            paper_url = clean_text(
                id_element.text
            )

        paper_id = paper_url.rstrip("/").split("/")[-1]

        # Remove version number:
        # 1706.03762v5 -> 1706.03762
        paper_id = re.sub(
            r"v\d+$",
            "",
            paper_id,
        )

        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------

        title_element = entry.find(
            "atom:title",
            namespace,
        )

        title = ""

        if title_element is not None:
            title = clean_text(
                title_element.text
            )

        # --------------------------------------------------------
        # Abstract
        # --------------------------------------------------------

        abstract_element = entry.find(
            "atom:summary",
            namespace,
        )

        abstract = ""

        if abstract_element is not None:
            abstract = clean_text(
                abstract_element.text
            )

        # --------------------------------------------------------
        # Authors
        # --------------------------------------------------------

        author_names = []

        for author in entry.findall(
            "atom:author",
            namespace,
        ):

            name_element = author.find(
                "atom:name",
                namespace,
            )

            if name_element is not None:

                name = clean_text(
                    name_element.text
                )

                if name:
                    author_names.append(name)

        authors = ", ".join(
            author_names
        )

        # --------------------------------------------------------
        # Categories
        # --------------------------------------------------------

        categories = []

        for category in entry.findall(
            "atom:category",
            namespace,
        ):

            term = category.attrib.get(
                "term",
                "",
            )

            if term:
                categories.append(term)

        categories_text = ", ".join(
            categories
        )

        # --------------------------------------------------------
        # Published date
        # --------------------------------------------------------

        published_element = entry.find(
            "atom:published",
            namespace,
        )

        published = ""

        if published_element is not None:

            published = clean_text(
                published_element.text
            )

        year = extract_year(
            published
        )

        # --------------------------------------------------------
        # Validate record
        # --------------------------------------------------------

        if not paper_id or not title:
            continue

        records.append(
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "categories": categories_text,
                "url": paper_url,
            }
        )

    df = pd.DataFrame(
        records,
        columns=[
            "paper_id",
            "title",
            "abstract",
            "authors",
            "year",
            "categories",
            "url",
        ],
    )

    if df.empty:

        print(
            "\nWARNING: arXiv API returned no valid papers."
        )

        return df

    # Remove duplicate IDs.
    df = df.drop_duplicates(
        subset=["paper_id"],
        keep="first",
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nValid unique papers: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# PAGINATED arXiv COLLECTION
# ============================================================

def collect_arxiv_dataset(
    search_query: str = "cat:cs.AI",
    total_papers: int = 1000,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Collect a larger arXiv dataset using multiple API requests.

    Example:

        collect_arxiv_dataset(
            search_query="cat:cs.AI",
            total_papers=1000
        )
    """

    all_frames = []

    current_start = 0

    while current_start < total_papers:

        remaining = (
            total_papers - current_start
        )

        current_batch = min(
            batch_size,
            remaining,
        )

        print("\n" + "-" * 70)

        print(
            f"Downloading papers "
            f"{current_start + 1} - "
            f"{current_start + current_batch}"
        )

        batch_df = fetch_arxiv_papers(
            search_query=search_query,
            max_results=current_batch,
            start=current_start,
        )

        if batch_df.empty:
            print(
                "No more papers returned by arXiv."
            )
            break

        all_frames.append(
            batch_df
        )

        current_start += len(batch_df)

        # Respect API request spacing.
        if current_start < total_papers:
            print(
                f"Waiting {REQUEST_DELAY} seconds..."
            )

            time.sleep(
                REQUEST_DELAY
            )

        # If fewer papers came back than requested,
        # there may be no more results.
        if len(batch_df) < current_batch:
            break

    if not all_frames:

        return pd.DataFrame(
            columns=[
                "paper_id",
                "title",
                "abstract",
                "authors",
                "year",
                "categories",
                "url",
            ]
        )

    final_df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    final_df = final_df.drop_duplicates(
        subset=["paper_id"],
        keep="first",
    )

    final_df = final_df.reset_index(
        drop=True
    )

    return final_df


# ============================================================
# OPTIONAL SEMANTIC SCHOLAR API
# ============================================================

def fetch_semantic_scholar_metadata(
    query: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Optional supplementary search using Semantic Scholar.

    This is NOT required for the main system.
    """

    params = {
        "query": query,
        "limit": limit,
        "fields": (
            "paperId,title,abstract,authors,"
            "year,fieldsOfStudy,url"
        ),
    }

    headers = {
        "User-Agent": USER_AGENT
    }

    try:

        response = requests.get(
            SEMANTIC_SCHOLAR_API_URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:

        print(
            f"Semantic Scholar API unavailable: {exc}"
        )

        return pd.DataFrame()

    records = []

    for paper in data.get(
        "data",
        [],
    ):

        paper_id = paper.get(
            "paperId"
        )

        title = paper.get(
            "title"
        )

        abstract = paper.get(
            "abstract"
        )

        if not paper_id or not title:
            continue

        authors_data = (
            paper.get("authors")
            or []
        )

        authors = ", ".join(
            author.get(
                "name",
                "",
            )
            for author in authors_data
        )

        fields = (
            paper.get(
                "fieldsOfStudy"
            )
            or []
        )

        records.append(
            {
                "paper_id": f"s2:{paper_id}",
                "title": title,
                "abstract": abstract or "",
                "authors": authors,
                "year": paper.get(
                    "year"
                ),
                "categories": ", ".join(
                    fields
                ),
                "url": paper.get(
                    "url",
                    "",
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# SAVE UNIFIED DATASET
# ============================================================

def save_dataset(
    arxiv_df: pd.DataFrame,
    output_file: str = OUTPUT_FILE,
    semantic_scholar_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine arXiv and optional Semantic Scholar data
    and save to CSV.
    """

    frames = [
        arxiv_df
    ]

    if (
        semantic_scholar_df is not None
        and not semantic_scholar_df.empty
    ):
        frames.append(
            semantic_scholar_df
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["paper_id"],
        keep="first",
    )

    combined = combined.reset_index(
        drop=True
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Total papers : {len(combined):,}"
    )

    print(
        f"Output file  : {output_file}"
    )

    return combined


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    try:

        # --------------------------------------------------------
        # CHANGE THIS QUERY FOR YOUR PROJECT
        # --------------------------------------------------------
        #
        # Useful examples:
        #
        # cat:cs.AI
        # cat:cs.LG
        # cat:cs.CL
        # all:machine learning
        # ti:"deep learning"
        #
        # For your project, we use AI/ML papers.
        # --------------------------------------------------------

        SEARCH_QUERY = "cat:cs.AI OR cat:cs.LG"

        TOTAL_PAPERS = 500

        BATCH_SIZE = 100

        print(
            "Starting arXiv API collection..."
        )

        arxiv_df = collect_arxiv_dataset(
            search_query=SEARCH_QUERY,
            total_papers=TOTAL_PAPERS,
            batch_size=BATCH_SIZE,
        )

        if arxiv_df.empty:

            raise RuntimeError(
                "No papers were collected from arXiv API."
            )

        # --------------------------------------------------------
        # Save dataset
        # --------------------------------------------------------

        save_dataset(
            arxiv_df=arxiv_df,
            output_file=OUTPUT_FILE,
        )

        print(
            "\nYou can now run:"
        )

        print(
            "    python preprocess.py"
        )

    except KeyboardInterrupt:

        print(
            "\nData collection stopped by user."
        )

    except Exception as exc:

        print(
            f"\nERROR: {exc}"
        )