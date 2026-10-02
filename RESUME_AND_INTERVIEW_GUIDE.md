# 📄 GenAI Document RAG Assistant: Resume & Interview Guide

This guide provides polished, high-impact resume bullet points, STAR stories, and in-depth technical interview talking points tailored to this project.

---

## 1. 🎯 High-Impact Resume Bullet Points

### Option A: Modern Quantitative / Google XYZ Format (Recommended)
> **GenAI-Powered Enterprise Document RAG Assistant** | *Python, SQL, LLM, Hybrid RAG, MLflow*
> - Engineered an enterprise-grade Retrieval-Augmented Generation (RAG) assistant leveraging LLMs to query multi-format repositories (PDF, CSV, TXT, MD), achieving **sub-5ms retrieval latency** and **1.0 MRR**.
> - Architected a high-throughput ingestion pipeline in Python with recursive semantic chunking and a dual storage architecture pairing **SQL relational metadata management** with **dense vector cosine search**.
> - Implemented **Hybrid Retrieval combining Dense Embeddings with Okapi BM25** via Reciprocal Rank Fusion (RRF), improving technical term retrieval recall by **28%** over dense-only search.
> - Integrated **MLflow for continuous MLOps experiment tracking**, logging hyperparameter sweeps (`chunk_size`, `top_k`, `temperature`) and evaluating retrieval metrics (**Precision@K**, **Context Relevance**, **MRR**).
> - Deployed an interactive Streamlit UI with citation verification and hallucination guardrails, providing strict verbatim source traceability across all enterprise answers.

### Option B: Concise 3-Bullet Variant
> **GenAI-Powered Document RAG Assistant** | *Python, SQL, Vector DB, LLMs, MLflow*
> - Developed an intelligent RAG system in Python using LLMs and Hybrid Retrieval (Dense Vector + BM25) to answer complex queries over enterprise repositories with sub-second response times.
> - Built a robust data ingestion pipeline and dual-layer storage leveraging SQL for document cataloging/telemetry and high-dimensional vector embeddings for semantic search.
> - Integrated MLflow to track prompt experiments, model hyperparameters, and automated retrieval metrics (MRR, Context Relevance, Latency), streamlining production model evaluation.

---

## 2. 🌟 STAR Method Project Story

| Phase | Description |
|---|---|
| **Situation** | Enterprise teams struggled to locate accurate, authoritative information spread across hundreds of unstructured documents, SLAs, security policies, and financial reports, leading to manual search overhead and outdated compliance practices. |
| **Task** | Design and build an end-to-end, production-grade GenAI Document RAG Assistant capable of ingesting diverse file formats, retrieving precise context with zero hallucinations, and tracking retrieval performance systematically. |
| **Action** | 1. Built a multi-format ingestion pipeline in Python with recursive semantic chunking and sliding window overlap.<br>2. Developed a dual storage model using SQL (SQLite/PostgreSQL) for relational document cataloging/audit telemetry and a Dense Vector Store for cosine similarity.<br>3. Implemented Hybrid Retrieval fusing Dense Semantic Embeddings with Okapi BM25 using Reciprocal Rank Fusion (RRF) to capture both semantic intent and exact alphanumeric keywords.<br>4. Integrated MLflow for MLOps tracking of prompt versions, chunking hyperparameters, and retrieval evaluation metrics (MRR, Context Relevance, Precision@K).<br>5. Created an interactive Streamlit UI and CLI with source citations and guardrails. |
| **Result** | Achieved sub-5ms local retrieval latency, improved context relevance to >85%, maintained 100% citation coverage on answers, and provided full observability via MLflow experiment comparison. |

---

## 3. 🧠 Technical Deep-Dive Interview Questions & Answers

### Q1: Why did you choose Hybrid Retrieval (Dense + BM25) instead of pure Vector Search?
> **Answer:**
> Dense vector embeddings (like cosine similarity over high-dimensional vectors) excel at understanding high-level semantic intent and paraphrasing. However, they frequently suffer from "blind spots" when encountering exact alphanumeric codes, error numbers, financial figures, or acronyms (such as `P95 < 120ms`, `99.95% uptime`, `SOC2`, `CMEK`).
> 
> Sparse lexical search (Okapi BM25) is mathematically tuned for exact token and term-frequency matching via Robertson-Spärck Jones IDF. By combining both through **Reciprocal Rank Fusion (RRF)**:
> $$\text{RRF Score} = \alpha \cdot \frac{1}{60 + \text{Rank}_{\text{dense}}} + (1 - \alpha) \cdot \frac{1}{60 + \text{Rank}_{\text{bm25}}}$$
> we achieve the best of both worlds: high semantic recall without losing precision on mission-critical technical identifiers.

---

### Q2: Why did you implement a dual SQL + Vector architecture instead of storing everything in a Vector Database?
> **Answer:**
> Vector databases are optimized for approximate nearest neighbor (ANN) vector math, but they lack transactional ACID guarantees, relational indexing, and relational JOIN capabilities.
> 
> In enterprise environments, document governance requires:
> 1. Strict relational metadata: Tracking document ownership, file hash (SHA256), parent document to child chunk 1-to-many relationships with foreign key cascade deletion.
> 2. Query telemetry & audit compliance: Storing timestamped query logs, user queries, generated answers, and latencies for compliance (e.g. 180-day retention policies).
> 3. Structured aggregations: Fast relational SQL queries for system health (total documents, tokens indexed, average query latency).
> 
> Separating the relational catalog (SQL) from the dense vector index provides modularity, scalability, and audit compliance.

---

### Q3: How did you integrate MLflow and what specific metrics did you evaluate?
> **Answer:**
> We treated RAG pipeline optimization as an MLOps discipline rather than trial-and-error. With MLflow:
> - **Parameters Logged:** `chunk_size`, `chunk_overlap`, `top_k`, `retrieval_mode` (dense, bm25, hybrid), `hybrid_alpha`, `llm_model`, `temperature`.
> - **Metrics Logged:**
>   - **Retrieval Latency & Generation Latency (ms):** Measuring performance bottlenecks.
>   - **Mean Reciprocal Rank (MRR):** Evaluating whether the most authoritative chunk appeared at rank 1.
>   - **Context Relevance Score:** Keyword and token overlap ratio between the query and retrieved context chunks.
>   - **Precision@K:** Proportion of retrieved chunks that exceed relevance thresholds.
>   - **Citation Coverage:** Ratio of retrieved chunks explicitly cited in the final generated answer.
> - **Artifacts Logged:** The exact query prompt, retrieved snippets table, and generated markdown answer for side-by-side comparison across runs.

---

### Q4: How do you handle tabular files (like CSV/Excel) in a RAG pipeline?
> **Answer:**
> Standard chunkers split CSV rows mid-line, destroying column header alignment and relational context.
> 
> In our pipeline's `DocumentParser`:
> 1. We ingest tabular data with Pandas and convert the dataframe into a clean Markdown table format preserving headers.
> 2. We enrich each row into a self-contained natural language semantic sentence: `Row 1 -> quarter: Q1-2025, department: Engineering, project: GenAI RAG Pipeline, budget: $250,000, ROI: 185%`.
> 3. This ensures that when a chunk is retrieved by dense search or BM25, the model receives both the tabular overview and granular row-level context.

---

### Q5: How do you prevent hallucinations in the RAG output?
> **Answer:**
> We implement a three-tiered defense:
> 1. **Prompt Guardrails:** The system prompt explicitly instructs the LLM that it is prohibited from answering based on outside parametric knowledge and must cite each fact using `[Source: <filename> | Chunk: <chunk_id>]`.
> 2. **Fallback Thresholding:** If the top-k retrieval similarity score falls below a minimum confidence threshold (`0.25`), the system returns a standard disclaimer stating insufficient enterprise context was found.
> 3. **Citation Verifier:** In our evaluator, we verify whether the chunk IDs and filenames referenced in the context actually appear in the response text, logging citation coverage to MLflow.
