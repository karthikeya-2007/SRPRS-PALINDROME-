# 📚 Scientific Research Paper Recommendation System

A **semantic research paper recommendation system** that helps users discover scientifically relevant papers based on the **meaning of their query or an existing research paper**, rather than relying only on keyword matching.

The system uses **Sentence-BERT embeddings** to convert research papers and queries into vector representations, then uses **FAISS similarity search** to retrieve the most semantically similar papers. A **Streamlit web interface** provides an interactive way to search and explore recommendations.

---

## 🚀 Features

- 🔎 **Semantic query search** — search using a research topic, question, or description.
- 📄 **Paper-to-paper recommendations** — select an existing paper and find similar papers.
- 🧠 **Sentence-BERT embeddings** using `all-MiniLM-L6-v2`.
- ⚡ **FAISS vector similarity search** for efficient retrieval.
- 📊 **Cosine similarity** for measuring semantic relevance.
- 📝 Displays paper title, authors, year, abstract, categories, similarity score, and paper URL.
- 🎚️ Configurable **Top-K recommendations** from 1 to 20.
- 💻 Interactive **Streamlit** interface.
- 📈 Built-in evaluation using **Precision@K, Recall@K, MRR, and response time**.

The Streamlit application exposes both **Query-based** and **Paper-based** search modes.

---

## 🧠 How It Works

The system follows this pipeline:

```text
                    ┌─────────────────────┐
                    │   arXiv API         │
                    │   Research Papers   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Collection     │
                    │ data_loader.py      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing       │
                    │ preprocess.py       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sentence-BERT       │
                    │ Embeddings          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FAISS Index         │
                    │ IndexFlatIP         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Search     │
                    │ search_engine.py    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Web App   │
                    │ app.py              │
                    └─────────────────────┘
```

---

# 🔄 System Pipeline

## 1. Data Collection

`data_loader.py` collects research papers from the **arXiv API**.

The default project configuration collects papers from:

```text
cat:cs.AI OR cat:cs.LG
```

with a target of **500 papers**, downloaded in batches of 100.

For every paper, the system collects:

- Paper ID
- Title
- Abstract
- Authors
- Publication year
- Categories
- URL

The arXiv API collector also removes duplicate paper IDs.

### Optional Semantic Scholar Support

The project also includes optional support for the **Semantic Scholar Academic Graph API**. This supplementary source is not required for the main recommendation pipeline.

---

## 2. Data Preprocessing

`preprocess.py` cleans and prepares the collected research papers.

The preprocessing pipeline performs:

- HTML entity decoding
- LaTeX cleanup
- Removal of mathematical delimiters
- Citation cleanup
- URL removal
- Special-character cleanup
- Whitespace normalization
- Duplicate removal
- Missing title/abstract filtering

The project creates a `combined_text` field by combining:

```text
title + [SEP] + abstract
```

This combined representation is used for generating semantic embeddings. 

---

## 3. Generate Semantic Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert research-paper text into numerical vector representations.

The embeddings are generated in batches and converted to NumPy arrays before indexing.

---

## 4. Normalize Embeddings

Before indexing, the embeddings are L2-normalized.

This allows the system to use **cosine similarity** through FAISS inner-product search.

---

## 5. FAISS Vector Index

The project uses:

```text
FAISS IndexFlatIP
```

to store and search the normalized embeddings.

```text
Research Paper
      ↓
Sentence-BERT
      ↓
Embedding Vector
      ↓
L2 Normalization
      ↓
FAISS IndexFlatIP
      ↓
Similarity Search
```

The generated index is saved as:

```text
paper_index.faiss
```

and the corresponding paper metadata is saved as:

```text
papers_metadata.pkl
```

The index builder stores the paper metadata alongside the vector index so search results can be mapped back to their original papers.

---

# 🔎 Search Modes

## Query-Based Search

Users can enter a research topic, question, or description such as:

```text
deep learning methods for medical image classification
```

The system converts the query into an embedding and searches the FAISS index for the most semantically similar papers.

Results include:

- Rank
- Title
- Authors
- Publication year
- Similarity score
- Abstract
- Categories
- Paper URL



---

## 📄 Paper-Based Search

Users can select an existing paper from the indexed dataset.

The selected paper's combined title and abstract representation is converted into an embedding. The system then retrieves other papers with similar semantic representations.

The selected source paper itself is excluded from the recommendations.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web application interface |
| **Sentence Transformers** | Semantic embeddings |
| **all-MiniLM-L6-v2** | Sentence-BERT model |
| **FAISS** | Vector similarity search |
| **Pandas** | Dataset processing |
| **NumPy** | Numerical operations |
| **scikit-learn** | Evaluation metrics |
| **Requests** | API communication |
| **arXiv API** | Research-paper data source |
| **Semantic Scholar API** | Optional supplementary source |

The application directly uses Streamlit, FAISS, and Sentence Transformers, while the data pipeline uses requests, pandas, and XML parsing for arXiv collection. 

---

# 📂 Project Structure

```text
SRPRS-PALINDROME-/
│
├── app.py
├── data_loader.py
├── preprocess.py
├── build_index.py
├── search_engine.py
├── evaluate.py
│
├── unified_papers.csv
├── processed_papers.csv
├── paper_index.faiss
├── papers_metadata.pkl
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `app.py` | Streamlit user interface |
| `data_loader.py` | Collects research papers from arXiv and optionally Semantic Scholar |
| `preprocess.py` | Cleans, validates, deduplicates, and prepares paper data |
| `build_index.py` | Generates embeddings and builds the FAISS index |
| `search_engine.py` | Provides semantic and similar-paper search functionality |
| `evaluate.py` | Evaluates retrieval performance |
| `unified_papers.csv` | Raw unified paper dataset |
| `processed_papers.csv` | Cleaned dataset used for indexing |
| `paper_index.faiss` | FAISS vector index |
| `papers_metadata.pkl` | Metadata corresponding to indexed papers |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/karthikeya-2007/SRPRS-PALINDROME-.git
cd SRPRS-PALINDROME-
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install pandas numpy requests faiss-cpu sentence-transformers streamlit scikit-learn
```

If the project contains a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The project follows a sequential pipeline.

## Step 1 — Collect Papers

Run:

```bash
python data_loader.py
```

This collects the configured arXiv dataset and creates:

```text
unified_papers.csv
```

The data loader is configured to collect AI and machine-learning papers by default.

---

## Step 2 — Preprocess the Dataset

Run:

```bash
python preprocess.py
```

This reads:

```text
unified_papers.csv
```

and creates:

```text
processed_papers.csv
```

The preprocessing stage also removes duplicate papers and records without usable titles or abstracts.

---

## Step 3 — Build the FAISS Index

Run:

```bash
python build_index.py
```

This process:

1. Loads the processed dataset.
2. Loads the Sentence-BERT model.
3. Generates embeddings.
4. Normalizes the embeddings.
5. Builds the FAISS `IndexFlatIP`.
6. Saves the vector index.
7. Saves paper metadata.



The generated files are:

```text
paper_index.faiss
papers_metadata.pkl
```

---

## Step 4 — Start the Web Application

Run:

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in your terminal.

The application automatically loads the Sentence-BERT model, FAISS index, and metadata using Streamlit resource caching.

---

# 🔬 Command-Line Search

The search engine can also be used directly without Streamlit.

Run:

```bash
python search_engine.py
```

You will be prompted:

```text
Enter a research topic:
```

Enter a topic such as:

```text
transformer models for natural language processing
```

The program returns the recommended papers and their similarity scores.

---

# 📊 Evaluation

The project includes an evaluation script:

```bash
python evaluate.py
```

The evaluation framework supports:

### Precision@K

Measures the proportion of retrieved papers that are relevant.

```text
Precision@K =
Relevant Retrieved Papers / K
```

### Recall@K

Measures how many relevant papers were retrieved.

```text
Recall@K =
Relevant Retrieved Papers / Total Relevant Papers
```

### MRR

**Mean Reciprocal Rank** evaluates how highly the first relevant paper appears in the results.

### Response Time

The system also measures search response time in milliseconds.



---

## ⚠️ Evaluation Note

The evaluation script currently supports manually labeled queries through `TEST_QUERIES`.

If no manual queries are supplied, it automatically generates **smoke-test queries from indexed paper titles**.

These smoke tests are useful for verifying that the retrieval pipeline works, but they are **not a research-quality benchmark**.

For meaningful evaluation, manually create a labeled test set containing:

```text
Query → Relevant Paper IDs
```

Then run:

```bash
python evaluate.py
```

---

# 📈 Retrieval Architecture

The core search operation is:

```text
User Query
    │
    ▼
Sentence-BERT
    │
    ▼
Query Embedding
    │
    ▼
L2 Normalization
    │
    ▼
FAISS IndexFlatIP
    │
    ▼
Top-K Similar Papers
    │
    ▼
Metadata Lookup
    │
    ▼
Ranked Results
```

The search engine normalizes the query embedding before passing it to FAISS, then returns ranked results containing metadata and similarity scores.

