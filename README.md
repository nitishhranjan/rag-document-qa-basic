# 📚 RAG Document QA

A production-ready Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions over a corpus of PDF documents. Built with a hybrid retrieval strategy (dense vector search + BM25 keyword search), ChromaDB for vector storage, and Groq-hosted Llama 3.1 as the LLM.

---

## Features

- **Hybrid Retrieval** — Combines dense semantic search (sentence-transformers) with sparse BM25 keyword search, weighted 70/30 via RRF fusion. Better recall than either method alone.
- **Conversation History** — Follow-up questions are automatically condensed into standalone queries for accurate retrieval, while the last 6 turns are injected into the LLM prompt for context-aware answers.
- **PDF Upload** — Upload any PDF at runtime to instantly extend the knowledge base (per-session, in-memory).
- **Unified Chat UI** — Single Streamlit app with document sidebar, file uploader, and chat interface with collapsible source citations.
- **ChromaDB Vector Store** — Persistent local vector database with cosine similarity. Pre-built index ships with the repo — no cold-start indexing on deploy.
- **Groq LLM (Llama 3.1 8B)** — Fast, low-latency inference via Groq's API.
- **Docker-ready** — Single `docker build` + `docker run` gets you running locally or on any Linux server.

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
├── app_unified.py            # Unified Streamlit UI (upload + chat, main entry point)
├── main.py                   # CLI entry point for testing
├── Dockerfile
├── requirements.txt
├── .env.example
├── data/
│   ├── pdf/                  # Source PDFs (xgboost, attention, embeddings, ...)
│   └── vector_store/         # Pre-built ChromaDB index (committed)
└── src/
    ├── config.py             # All settings (paths, model names, chunk sizes)
    ├── document_loader.py    # PDF → LangChain Documents
    ├── chunker.py            # RecursiveCharacterTextSplitter
    ├── embeddings.py         # SentenceTransformer wrapper
    ├── vectorstore.py        # ChromaDB CRUD
    ├── rag_pipeline.py       # Prompt template, conversation history, LLM call
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
streamlit run app_unified.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker (local)

```bash
docker build -t rag-qa .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here rag-qa
```

Open [http://localhost:8000](http://localhost:8000).

---

## Deploy to AWS EC2 (free tier)

This app runs on a **t3.micro** instance (1 vCPU, 1 GB RAM). Swap space is required to load the embedding model without OOM.

---

## Tech Stack


| Layer            | Technology                                                          |
| ---------------- | ------------------------------------------------------------------- |
| UI               | [Streamlit](https://streamlit.io)                                   |
| Vector store     | [ChromaDB](https://www.trychroma.com)                               |
| Embeddings       | [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| Keyword search   | [rank-bm25](https://github.com/dorianbrown/rank_bm25)               |
| LLM              | [Groq](https://groq.com) (Llama 3.1 8B Instant)                     |
| Document parsing | [PyMuPDF](https://pymupdf.readthedocs.io)                           |
| Orchestration    | [LangChain](https://langchain.com)                                  |
| Containerization | Docker                                                              |
| Cloud            | AWS EC2 (t3.micro)                                                  |


---

## License

MIT