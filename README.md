# 📚 RAG Document QA

A production-ready Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions over a corpus of PDF documents. Built with a hybrid retrieval strategy (dense vector search + BM25 keyword search), ChromaDB for vector storage, and Groq-hosted Llama 3.1 as the LLM.

---

## Features

- **Hybrid Retrieval** — Combines dense semantic search (sentence-transformers) with sparse BM25 keyword search, weighted 70/30 by default. Better recall than either method alone.
- **Streamlit UI** — Clean search interface with source citations, relevance scores, and configurable top-k retrieval.
- **ChromaDB Vector Store** — Persistent local vector database with cosine similarity. Pre-built index ships with the repo — no cold-start indexing on deploy.
- **Groq LLM (Llama 3.1 8B)** — Fast, low-latency inference via Groq's API.
- **Docker-ready** — Single `docker build` + `docker run` gets you running locally or on any container platform.

---

## Architecture

```
PDF Documents
      │
      ▼
┌─────────────┐
│  PDF Loader │  (PyMuPDF / pypdf)
└──────┬──────┘
       │ raw text
       ▼
┌─────────────┐
│   Chunker   │  chunk_size=1000, overlap=200
└──────┬──────┘
       │ chunks
       ┌──────────────────────────┐
       │                          │
       ▼                          ▼
┌─────────────┐          ┌─────────────────┐
│  Embeddings │          │  BM25 Index     │
│ (MiniLM-L6) │          │  (rank-bm25)    │
└──────┬──────┘          └────────┬────────┘
       │ vectors                  │ term scores
       ▼                          │
┌─────────────┐                   │
│  ChromaDB   │                   │
│ VectorStore │                   │
└──────┬──────┘                   │
       │                          │
       └──────────┬───────────────┘
                  ▼
        ┌──────────────────┐
        │ HybridRetriever  │  RRF fusion (α=0.7)
        └────────┬─────────┘
                 │ top-k chunks + scores
                 ▼
        ┌──────────────────┐
        │   RAG Pipeline   │  prompt + context
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Groq / Llama 3.1│
        └────────┬─────────┘
                 │
                 ▼
           Final Answer
         + Source Citations
```

---

## Project Structure

```
rag-document-qa-basic/
├── app.py                    # Streamlit search UI (main entry point)
├── app_chat.py               # Streamlit chat UI (alternative)
├── main.py                   # CLI entry point for testing
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
├── data/
│   ├── pdf/                  # Source PDFs (xgboost, attention, embeddings, ...)
│   ├── text_files/           # Optional plain-text sources
│   └── vector_store/         # Pre-built ChromaDB index (committed)
└── src/
    ├── config.py             # All settings (paths, model names, chunk sizes)
    ├── document_loader.py    # PDF → LangChain Documents
    ├── chunker.py            # RecursiveCharacterTextSplitter
    ├── embeddings.py         # SentenceTransformer wrapper
    ├── vectorstore.py        # ChromaDB CRUD
    ├── rag_pipeline.py       # Prompt template + LLM call
    └── retrievers/
        ├── vector_retriever.py
        ├── bm25_retriever.py
        └── hybrid_retriever.py   # RRF fusion
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone & install

```bash
git clone https://github.com/your-username/rag-document-qa-basic.git
cd rag-document-qa-basic

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Tech Stack


| Layer            | Technology                                                                        |
| ---------------- | --------------------------------------------------------------------------------- |
| UI               | [Streamlit](https://streamlit.io)                                                 |
| Vector store     | [ChromaDB](https://www.trychroma.com)                                             |
| Embeddings       | [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`)               |
| Keyword search   | [rank-bm25](https://github.com/dorianbrown/rank_bm25)                             |
| LLM              | [Groq](https://groq.com) (Llama 3.1 8B Instant)                                   |
| Document parsing | [PyMuPDF](https://pymupdf.readthedocs.io) + [pypdf](https://pypdf.readthedocs.io) |
| Orchestration    | [LangChain](https://langchain.com)                                                |
| Containerization | Docker                                                                            |
| Cloud            | AWS App Runner + ECR                                                              |


---

## License

MIT