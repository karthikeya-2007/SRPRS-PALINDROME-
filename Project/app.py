
from __future__ import annotations

import time
from pathlib import Path

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

from search_engine import (
    MODEL_NAME,
    embed_text,
    find_similar_with_resources,
    search_with_resources,
)


# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Scientific Paper Recommendation",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Load Sentence-BERT once and keep it in Streamlit cache.
    """
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_faiss_index():
    """
    Load the FAISS index once.
    """

    index_path = Path("paper_index.faiss")

    if not index_path.exists():
        raise FileNotFoundError(
            "paper_index.faiss not found. "
            "Run build_index.py first."
        )

    return faiss.read_index(
        str(index_path)
    )


@st.cache_resource
def load_metadata():
    """
    Load the metadata file once.
    """

    import pickle

    metadata_path = Path(
        "papers_metadata.pkl"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "papers_metadata.pkl not found. "
            "Run build_index.py first."
        )

    with open(metadata_path, "rb") as file:
        return pickle.load(file)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("⚙️ Search Settings")

top_k = st.sidebar.slider(
    "Top-K recommendations",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)

search_mode = st.sidebar.radio(
    "Search mode",
    options=[
        "Query-based",
        "Paper-based",
    ],
)

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")

st.sidebar.write(
    """
**Primary source:** arXiv Kaggle dataset

**Optional source:** Semantic Scholar Academic Graph API

**Embedding model:**  
`all-MiniLM-L6-v2`

**Vector database:**  
FAISS `IndexFlatIP`

**Similarity:**  
Cosine similarity
"""
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Semantic search recommends papers based on "
    "the meaning of the query/paper rather than "
    "only matching keywords."
)


# ---------------------------------------------------------------------
# Load system resources
# ---------------------------------------------------------------------

try:
    model = load_model()
    index = load_faiss_index()
    metadata = load_metadata()

except FileNotFoundError as exc:

    st.error(
        f"⚠️ {exc}"
    )

    st.info(
        """
Run the following commands in order:

1. `python data_loader.py`
2. `python preprocess.py`
3. `python build_index.py`
4. `streamlit run app.py`
"""
    )

    st.stop()

except Exception as exc:

    st.error(
        f"Could not load the recommendation system: {exc}"
    )

    st.stop()


# ---------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------

st.title(
    "📚 Scientific Research Paper Recommendation"
)

st.write(
    "Semantic Research Paper Recommendation Using "
    "Sentence Embeddings and FAISS"
)

st.markdown(
    """
The system converts research papers into semantic vector
representations using Sentence-BERT and retrieves the most
semantically similar papers using FAISS.
"""
)


# ---------------------------------------------------------------------
# Query-based search
# ---------------------------------------------------------------------

if search_mode == "Query-based":

    st.subheader(
        "🔎 Search Research Papers"
    )

    query = st.text_input(
        "Enter a research topic, question, or description",
        placeholder=(
            "Example: deep learning methods for "
            "medical image classification"
        ),
    )

    search_button = st.button(
        "🔍 Find Papers",
        type="primary",
        use_container_width=False,
    )

    if search_button:

        if not query.strip():

            st.warning(
                "Please enter a research query."
            )

        else:

            start_time = time.perf_counter()

            try:

                results = search_with_resources(
                    query=query,
                    top_k=top_k,
                    model=model,
                    index=index,
                    metadata=metadata,
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                st.caption(
                    f"Search response time: "
                    f"{elapsed * 1000:.2f} ms"
                )

                if not results:

                    st.warning(
                        "No matching papers were found."
                    )

                else:

                    st.success(
                        f"Found {len(results)} recommendations."
                    )

                    for result in results:

                        st.markdown(
                            f"### {result['rank']}. "
                            f"{result['title']}"
                        )

                        # Metadata row.
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write(
                                f"**Authors:** "
                                f"{result['authors'] or 'Not available'}"
                            )

                        with col2:
                            st.write(
                                f"**Year:** "
                                f"{result['year'] or 'Unknown'}"
                            )

                        with col3:
                            st.write(
                                f"**Similarity:** "
                                f"{result['similarity_score']:.4f}"
                            )

                        # Abstract.
                        with st.expander(
                            "View Abstract"
                        ):
                            st.write(
                                result["abstract"]
                                or "Abstract not available."
                            )

                        # Category.
                        if result.get("categories"):
                            st.caption(
                                f"Categories: "
                                f"{result['categories']}"
                            )

                        # Reference link.
                        if result.get("url"):
                            st.markdown(
                                f"[🔗 Open Paper]({result['url']})"
                            )

                        st.markdown("---")

            except Exception as exc:

                st.error(
                    f"Search failed: {exc}"
                )


# ---------------------------------------------------------------------
# Paper-based recommendation
# ---------------------------------------------------------------------

else:

    st.subheader(
        "📄 Find Similar Papers"
    )

    if not metadata:

        st.warning(
            "No papers are available in the index."
        )

    else:

        # Build readable labels while keeping IDs internally.
        paper_options = {}

        for item in metadata:

            paper_id = str(
                item.get("paper_id", "")
            )

            title = str(
                item.get("title", "")
            )

            if paper_id and title:

                label = (
                    f"{title[:100]} "
                    f"({paper_id})"
                )

                paper_options[label] = paper_id

        selected_label = st.selectbox(
            "Select a paper",
            options=list(
                paper_options.keys()
            ),
        )

        selected_paper_id = paper_options[
            selected_label
        ]

        # Display the selected paper.
        selected_paper = next(
            (
                item
                for item in metadata
                if str(
                    item.get("paper_id", "")
                ) == selected_paper_id
            ),
            None,
        )

        if selected_paper:

            st.markdown(
                f"**Selected Paper:** "
                f"{selected_paper['title']}"
            )

            if selected_paper.get("authors"):
                st.caption(
                    f"Authors: "
                    f"{selected_paper['authors']}"
                )

            if selected_paper.get("abstract"):

                with st.expander(
                    "View Selected Paper Abstract"
                ):
                    st.write(
                        selected_paper[
                            "abstract"
                        ]
                    )

        find_button = st.button(
            "🔗 Find Similar Papers",
            type="primary",
        )

        if find_button:

            start_time = time.perf_counter()

            try:

                results = find_similar_with_resources(
                    paper_id=selected_paper_id,
                    top_k=top_k,
                    model=model,
                    index=index,
                    metadata=metadata,
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                st.caption(
                    f"Search response time: "
                    f"{elapsed * 1000:.2f} ms"
                )

                if not results:

                    st.warning(
                        "No similar papers were found."
                    )

                else:

                    st.success(
                        f"Found {len(results)} similar papers."
                    )

                    for result in results:

                        st.markdown(
                            f"### {result['rank']}. "
                            f"{result['title']}"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write(
                                f"**Authors:** "
                                f"{result['authors'] or 'Not available'}"
                            )

                        with col2:
                            st.write(
                                f"**Year:** "
                                f"{result['year'] or 'Unknown'}"
                            )

                        with col3:
                            st.write(
                                f"**Similarity:** "
                                f"{result['similarity_score']:.4f}"
                            )

                        with st.expander(
                            "View Abstract"
                        ):
                            st.write(
                                result["abstract"]
                                or "Abstract not available."
                            )

                        if result.get("categories"):
                            st.caption(
                                f"Categories: "
                                f"{result['categories']}"
                            )

                        if result.get("url"):
                            st.markdown(
                                f"[🔗 Open Paper]({result['url']})"
                            )

                        st.markdown("---")

            except ValueError as exc:

                st.error(
                    str(exc)
                )

            except Exception as exc:

                st.error(
                    f"Recommendation failed: {exc}"
                )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown("---")

st.caption(
    "Scientific Research Paper Recommendation System | "
    "Sentence-BERT + FAISS + Streamlit"
)