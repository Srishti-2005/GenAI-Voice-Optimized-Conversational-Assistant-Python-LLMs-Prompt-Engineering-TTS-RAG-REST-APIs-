# ⚡ GenAI-Powered Document RAG Assistant
### Python • SQL • LLM • Hybrid RAG • MLflow

A production-grade, enterprise-ready **Retrieval-Augmented Generation (RAG)** system designed to ingest heterogeneous enterprise documents, maintain relational metadata in SQL, perform low-latency hybrid semantic search, generate grounded responses with citations, and track experiments and retrieval metrics with MLflow.

---

## 🌟 Key Architecture & Highlights

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION PIPELINE                         │
│  PDF, TXT, Markdown, CSV, JSON  ──>  Parser  ──>  Semantic Chunker    │
└───────────────────┬─────────────────────────────────┬──────────────────┘
                    │                                 │
                    ▼                                 ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│      SQL RELATIONAL CATALOG     │   │      VECTOR RETRIEVAL STORE      │
│  - documents (Metadata & Stats) │   │  - Dense Semantic Embeddings     │
│  - document_chunks (Parent/FK)  │   │  - Sub-millisecond Cosine Matrix │
│  - query_logs (Telemetry audit) │   │  - Okapi BM25 Lexical Keyword    │
│  - eval_metrics (MRR, Rel.)     │   │  - Reciprocal Rank Fusion (RRF)  │
└───────────────────┬─────────────┘   └───────────────┬──────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   GROUNDED GENERATION & EXPERIMENTATION                │
│  Prompt Engineering Template  ──>  Unified LLM Client (Cloud/Offline)  │
│  Citation Verifier            ──>  RAGEvaluator (Relevance, MRR)      │
│  MLflow Experiment Tracker    ──>  Params, Metrics & Artifacts Logs    │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Multi-Format Python Ingestion Pipeline**:
   - Parses `.pdf`, `.txt`, `.md`, `.csv` (markdown table & row representations), and `.json`.
   - Recursive character chunking preserving sentence boundaries and configurable sliding overlap.
   - Deterministic SHA256 document hashing and metadata enrichment.

2. **Dual-Store Architecture (SQL + Vector Search)**:
   - **SQL Layer (SQLite / PostgreSQL)**: Enforces relational schema across documents, chunks, query telemetry, and evaluation metrics with foreign key integrity.
   - **Dense Vector Store**: Normalized float32 embedding matrix with cosine similarity dot product.
   - **Lexical BM25 Index**: Robertson-Spärck Jones Okapi BM25 algorithm with dynamic IDF smoothing to capture technical identifiers (e.g. `P95`, `99.95%`, `SOC2`, `CMEK`).

3. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - Fuses dense semantic vector rankings and sparse BM25 scores:
     $$\text{RRF Score} = \alpha \cdot \frac{1}{60 + \text{Rank}_{\text{dense}}} + (1 - \alpha) \cdot \frac{1}{60 + \text{Rank}_{\text{bm25}}}$$
   - Eliminates dense embedding blind spots for exact alphanumeric codes.

4. **Hallucination Prevention & Citation Grounding**:
   - Strict citation enforcement format: `[Source: <filename> | Chunk: <chunk_id>]`.
   - Automated fallback guardrails when context relevance falls below threshold.

5. **MLflow Experiment & Retrieval Metric Tracking**:
   - Tracks hyperparameter sweeps: `chunk_size`, `chunk_overlap`, `top_k`, `retrieval_mode`, `hybrid_alpha`, `llm_model`, `temperature`.
   - Evaluates retrieval metrics per query: **Context Relevance**, **Mean Reciprocal Rank (MRR)**, **Precision@K**, **Retrieval Latency (ms)**, and **Generation Latency (ms)**.
   - Saves query prompts, retrieved context tables, and response markdown artifacts directly to MLflow.

6. **Pluggable Multi-Provider LLM & Embeddings**:
   - Built-in **Zero-Cost Offline Extractive Engine** (runs immediately out-of-the-box without API keys).
   - Instant plug-and-play support for **OpenAI** (`gpt-4o-mini`, `gpt-4o`), **Google Gemini**, and local **Ollama** models.

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Automated End-to-End Demo
Populates sample enterprise documents, executes hybrid queries, prints performance metrics, and logs runs to MLflow:
```bash
python run_demo.py
```

### 3. Launch the Streamlit Interactive UI
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore:
- **Assistant & Query Studio**: Interactive chat with citation inspection and confidence metrics.
- **Ingestion & SQL Catalog**: Document uploader, SQL catalog table, and chunk explorer.
- **MLflow Experiment Tracking**: Run history, latency graphs, and parameter comparisons.
- **Architecture & Specs**: Interactive pipeline diagram and configurations.

### 4. Launch the MLflow Dashboard
Explore the full interactive MLflow dashboard:
```bash
python -m mlflow ui --backend-store-uri sqlite:///data/storage/mlflow.db --port 5000
```
Open [http://localhost:5000](http://localhost:5000) to view logged parameters, metrics, run comparisons, and execution artifacts.

---

## 💻 CLI Usage

The assistant provides a command line interface:

```bash
# Ingest documents from a directory
python -m rag_assistant.cli ingest --path data/sample_docs

# Query using Hybrid retrieval
python -m rag_assistant.cli query "What is the platform monthly uptime guarantee?" --mode hybrid --top-k 4

# Query using Dense Vector retrieval only
python -m rag_assistant.cli query "What are the latency SLAs for P95 API gateway?" --mode dense

# Display SQL and Vector DB telemetry
python -m rag_assistant.cli stats

# Run automated multi-query benchmark
python -m rag_assistant.cli benchmark
```

---

## 🧪 Running the Test Suite

Run the unit and integration test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

All tests validate:
- Multi-format parsing (PDF, TXT, CSV)
- Recursive sliding-window chunking
- SQL relational schema & foreign key cascades
- Dense vector similarity dot product
- BM25 lexical ranking
- End-to-end RAG pipeline execution & MLflow logging

---

## ⚙️ Configuration (.env)

Customize parameters in `.env`:
```env
# Optional: External Cloud LLMs (Leave blank for offline engine)
OPENAI_API_KEY=
GEMINI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# MLflow Settings
MLFLOW_EXPERIMENT_NAME=Enterprise-Document-RAG
MLFLOW_TRACKING_URI=sqlite:///data/storage/mlflow.db

# RAG Pipeline Defaults
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=80
DEFAULT_TOP_K=4
DEFAULT_RETRIEVAL_MODE=hybrid
```

---

## 📁 Repository Structure

```
├── data/
│   ├── sample_docs/                  # Enterprise documents (Policy, SLA, Arch, Financial CSV)
│   └── storage/                      # SQLite database, vector index, and MLflow backend
├── rag_assistant/
│   ├── ingestion/
│   │   ├── parser.py                 # Multi-format document parser (PDF, CSV, TXT, MD, JSON)
│   │   └── chunker.py                # Recursive character chunker with sliding overlap
│   ├── storage/
│   │   ├── sql_store.py              # Relational SQL store (documents, chunks, query telemetry)
│   │   └── vector_store.py           # Dense vector index with cosine similarity search
│   ├── retrieval/
│   │   ├── bm25.py                   # Okapi BM25 sparse keyword search engine
│   │   ├── embeddings.py             # Local semantic & cloud embedding adapters
│   │   └── hybrid_retriever.py       # Hybrid retrieval with Reciprocal Rank Fusion (RRF)
│   ├── generation/
│   │   ├── prompt_templates.py       # Grounded prompt engineering with citation rules
│   │   └── llm_client.py             # Multi-provider LLM adapter (Offline, OpenAI, Gemini, Ollama)
│   ├── tracking/
│   │   ├── evaluator.py              # Retrieval metrics (Context Relevance, MRR, Precision@K)
│   │   └── mlflow_tracker.py         # Automated MLflow experiment & artifact tracking
│   ├── pipeline.py                   # Unified end-to-end RAG orchestrator
│   └── cli.py                        # Terminal CLI commands
├── tests/                            # Unit and integration test suite
├── app.py                            # Streamlit web application
├── run_demo.py                       # Automated end-to-end demo runner
├── requirements.txt                  # Python dependencies
└── RESUME_AND_INTERVIEW_GUIDE.md     # Resume bullet points & interview prep guide
```
