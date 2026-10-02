import streamlit as st
import os
import time
import pandas as pd
from pathlib import Path
from rag_assistant.config import RAGConfig
from rag_assistant.pipeline import RAGPipeline

# Page Configuration
st.set_page_config(
    page_title="GenAI Enterprise Document RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #151522 100%);
        padding: 24px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .badge-rag { background: #6366f1; color: white; }
    .badge-sql { background: #06b6d4; color: white; }
    .badge-mlflow { background: #059669; color: white; }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card .num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    .chunk-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    cfg = RAGConfig()
    return RAGPipeline(config=cfg)

pipeline = get_pipeline()

# Sidebar: Controls & Live Settings
with st.sidebar:
    st.markdown("### ⚙️ RAG Engine Parameters")
    
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Hybrid combines Dense Vector Cosine Similarity and Sparse BM25 via Reciprocal Rank Fusion (RRF)."
    )
    
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=4)
    
    if retrieval_mode == "hybrid":
        alpha = st.slider(
            "Dense Vector Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="1.0 = Pure Vector Search, 0.0 = Pure BM25 Lexical Keyword Search."
        )
    else:
        alpha = 0.6
        
    st.markdown("---")
    st.markdown("### 🤖 LLM & Model Engine")
    
    provider_choice = st.selectbox(
        "Inference Provider",
        options=["offline (Zero API Cost)", "openai", "gemini", "ollama"],
        index=0,
    )
    clean_provider = provider_choice.split()[0].lower()
    
    if clean_provider == "openai":
        api_key_input = st.text_input("OpenAI API Key", type="password", value=pipeline.config.openai_api_key)
        model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        if api_key_input:
            pipeline.config.openai_api_key = api_key_input
            pipeline.config.llm_provider = "openai"
            pipeline.config.llm_model = model_name
            pipeline.llm_client = pipeline.llm_client.__class__.__subclasses__()[0](api_key=api_key_input, model=model_name)
    elif clean_provider == "gemini":
        gemini_key = st.text_input("Gemini API Key", type="password", value=pipeline.config.gemini_api_key)
        if gemini_key:
            pipeline.config.gemini_api_key = gemini_key
            pipeline.config.llm_provider = "gemini"
    else:
        pipeline.config.llm_provider = "offline"
        st.info("💡 Using built-in high-precision extractive LLM engine with citation grounding.")

    st.markdown("---")
    stats = pipeline.get_stats()
    st.markdown("### 📊 System Telemetry")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("SQL Documents", stats.get("total_docs", 0))
        st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
    with col_s2:
        st.metric("Vector Chunks", stats.get("total_chunks", 0))
        st.metric("Queries Logged", stats.get("total_queries", 0))


# Header Banner
st.markdown("""
<div class="main-header">
    <div style="margin-bottom: 8px;">
        <span class="badge-pill badge-rag">RAG Pipeline</span>
        <span class="badge-pill badge-sql">SQL Telemetry</span>
        <span class="badge-pill badge-mlflow">MLflow Tracked</span>
    </div>
    <h2 style="margin: 0; color: #ffffff;">GenAI-Powered Document RAG Assistant</h2>
    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 0.95rem;">
        Intelligent multi-format document retrieval, structured SQL metadata catalog, hybrid semantic vector search, and automated MLflow retrieval metrics tracking.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Assistant & Query Studio",
    "📂 Ingestion & SQL Catalog",
    "📈 MLflow Experiment Tracking",
    "🏗️ Architecture & Specs"
])

# -------------------------------------------------------------
# TAB 1: Query Studio
# -------------------------------------------------------------
with tab1:
    st.markdown("#### 🔍 Ask Contextual Queries Across Enterprise Repositories")
    
    # Pre-canned queries for fast evaluation
    st.markdown("**Sample Enterprise Questions (Click to test):**")
    sample_queries = [
        "What is the platform uptime guarantee and service credit calculation?",
        "What are the mandatory security guidelines for external LLM API usage and PII?",
        "What are the disaster recovery RTO and RPO targets in the cloud architecture spec?",
        "What was the allocated budget, actual spend, and ROI for the GenAI RAG Pipeline?",
    ]
    
    cols = st.columns(len(sample_queries))
    selected_sample = None
    for i, q in enumerate(sample_queries):
        if cols[i].button(f"📌 Query {i+1}", help=q, use_container_width=True):
            selected_sample = q

    default_query = selected_sample or "What is the platform uptime guarantee and service credit calculation?"
    user_query = st.text_area("Enter your query:", value=default_query, height=85)
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_query = st.button("🚀 Run RAG Query", type="primary", use_container_width=True)
        
    if run_query and user_query.strip():
        with st.spinner("Retrieving relevant context and generating grounded answer..."):
            result = pipeline.query(
                query_text=user_query.strip(),
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                alpha=alpha,
            )
            
            # Response Display
            st.markdown("### 📝 Grounded Response")
            st.markdown(result["answer"])
            
            # Latency and Evaluation Metrics Bar
            eval_res = result["eval_result"]
            latency = result["latency"]
            
            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="num">{latency['total_ms']}ms</div>
                    <div class="label">Total Latency</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="num">{latency['retrieval_ms']}ms</div>
                    <div class="label">Retrieval Latency</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="num">{eval_res.context_relevance * 100:.1f}%</div>
                    <div class="label">Context Relevance</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="num">{eval_res.avg_similarity:.3f}</div>
                    <div class="label">Avg Similarity</div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="num">{eval_res.mrr_score:.2f}</div>
                    <div class="label">MRR Score</div>
                </div>
                """, unsafe_allow_html=True)
                
            if result.get("mlflow_run_id"):
                st.caption(f"✅ Experiment run automatically logged to MLflow (Run ID: `{result['mlflow_run_id']}`)")
            
            # Retrieved Chunks Accordion
            st.markdown("#### 📚 Retrieved Context Snippets & Citations")
            for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                score = chunk.get("final_score", chunk.get("similarity_score", 0.0))
                with st.expander(f"Snippet #{idx}: {chunk.get('filename')} (ID: {chunk.get('chunk_id')}) — Score: {score:.4f}", expanded=(idx == 1)):
                    st.markdown(f"**Source Document:** `{chunk.get('filename')}` | **Token Count:** {chunk.get('token_count')}")
                    st.markdown(f"""<div class="chunk-box">{chunk.get('content')}</div>""", unsafe_allow_html=True)
                    if "dense_rank" in chunk or "bm25_rank" in chunk:
                        st.caption(f"Dense Rank: {chunk.get('dense_rank')} | BM25 Rank: {chunk.get('bm25_rank')} | Combined RRF: {chunk.get('final_score'):.5f}")

# -------------------------------------------------------------
# TAB 2: Ingestion & SQL Catalog
# -------------------------------------------------------------
with tab2:
    st.markdown("#### 📥 Document Ingestion Pipeline & Relational Data Management")
    
    col_up, col_actions = st.columns([2, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload Enterprise Documents (PDF, TXT, MD, CSV, JSON)",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if st.button("Process & Index Uploaded Documents", type="primary"):
                for up_file in uploaded_files:
                    bytes_data = up_file.getvalue()
                    res = pipeline.ingest_bytes(up_file.name, bytes_data)
                    st.success(f"Indexed `{res['filename']}`: {res['total_chunks']} chunks, {res['total_tokens']} tokens.")
                st.rerun()

    with col_actions:
        st.markdown("##### ⚡ Quick Ingest Sample Corpus")
        st.write("Ingest the pre-bundled enterprise security, architecture, legal SLA, and financial documents.")
        if st.button("Ingest Sample Documents", use_container_width=True):
            sample_dir = Path("data/sample_docs")
            with st.spinner("Ingesting sample documents into SQL and Vector index..."):
                results = pipeline.ingest_directory(sample_dir)
                st.success(f"Ingested {len(results)} sample enterprise documents!")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.markdown("#### 🗄️ SQL Document Catalog (`documents` table)")
    docs = pipeline.sql_store.get_all_documents()
    if docs:
        df_docs = pd.DataFrame(docs)
        st.dataframe(
            df_docs[["doc_id", "filename", "file_type", "file_size_bytes", "total_chunks", "total_tokens", "created_at"]],
            use_container_width=True,
        )
    else:
        st.warning("No documents indexed yet. Click 'Ingest Sample Documents' above to populate the repository!")

    st.markdown("#### 🧩 Relational Chunk Explorer (`document_chunks` table)")
    if docs:
        doc_options = {d["doc_id"]: f"{d['filename']} ({d['total_chunks']} chunks)" for d in docs}
        sel_doc_id = st.selectbox("Inspect Chunks for Document:", options=list(doc_options.keys()), format_func=lambda x: doc_options[x])
        chunks = pipeline.sql_store.get_chunks_by_doc_id(sel_doc_id)
        if chunks:
            df_chunks = pd.DataFrame(chunks)
            st.dataframe(df_chunks[["chunk_id", "chunk_index", "token_count", "content"]], use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MLflow Experiment Tracking
# -------------------------------------------------------------
with tab3:
    st.markdown("#### 📈 MLflow Experiment Tracking & Retrieval Metrics")
    st.write(
        "Every query executed in this system tracks hyperparameter configurations (`chunk_size`, `retrieval_mode`, `top_k`, `temperature`) "
        "and retrieval evaluation metrics (`latency`, `context_relevance`, `MRR`, `precision@k`) in MLflow."
    )
    
    st.info(f"**MLflow Tracking URI:** `{pipeline.config.mlflow_tracking_uri}` | **Experiment:** `{pipeline.config.mlflow_experiment_name}`")
    
    if pipeline.mlflow_tracker:
        runs_df = pipeline.mlflow_tracker.get_runs_dataframe(max_results=30)
        if not runs_df.empty:
            st.markdown("##### 🧪 Recent Experiment Runs")
            
            # Select key columns for display
            metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
            param_cols = [c for c in runs_df.columns if c.startswith("params.")]
            tag_cols = [c for c in runs_df.columns if c in ("tags.query_preview", "tags.retrieval_mode")]
            display_cols = ["run_id", "start_time"] + tag_cols + metric_cols + param_cols
            display_cols = [c for c in display_cols if c in runs_df.columns]
            
            st.dataframe(runs_df[display_cols], use_container_width=True)
            
            # Metric plots
            st.markdown("##### 📊 Performance Visualizations")
            col_p1, col_p2 = st.columns(2)
            
            if "metrics.total_latency_ms" in runs_df.columns:
                with col_p1:
                    st.write("**Total Latency per Run (ms)**")
                    st.line_chart(runs_df["metrics.total_latency_ms"].dropna())
                    
            if "metrics.eval_context_relevance" in runs_df.columns:
                with col_p2:
                    st.write("**Context Relevance Score**")
                    st.bar_chart(runs_df["metrics.eval_context_relevance"].dropna())
        else:
            st.warning("No MLflow runs logged yet. Execute queries in the 'Assistant & Query Studio' tab to generate tracking data!")
            
    st.markdown("---")
    st.markdown("##### 🖥️ How to Launch the MLflow Web Dashboard")
    st.code("python -m mlflow ui --backend-store-uri sqlite:///data/storage/mlflow.db --port 5000", language="bash")
    st.caption("Run this command in your terminal to explore the full interactive MLflow comparison dashboard, artifact viewer, and parameter charts at http://localhost:5000.")

# -------------------------------------------------------------
# TAB 4: Architecture & Specs
# -------------------------------------------------------------
with tab4:
    st.markdown("#### 🏗️ System Architecture & Engineering Specifications")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Multi-Format Documents PDF, CSV, TXT, MD, JSON] --> B[DocumentParser]
        B --> C[DocumentChunker: Semantic Overlap]
        C --> D[(SQL Catalog: SQLite/PostgreSQL)]
        C --> E[Local / Dense Embedder]
        E --> F[(Dense Vector Store)]
        C --> G[Okapi BM25 Index]
        
        H[User Query] --> I[HybridRetriever]
        F -->|Dense Cosine Similarity| I
        G -->|Sparse Lexical Matching| I
        I -->|Reciprocal Rank Fusion RRF| J[Ranked Context Chunks]
        
        J --> K[Prompt Engineering & Citation Grounding]
        K --> L[LLM Inference Engine]
        L --> M[Structured Grounded Response]
        
        M --> N[RAGEvaluator: MRR, Relevance, Latency]
        N --> O[(MLflow Experiment Tracking)]
        N --> D
    ```
    """)
    
    st.markdown("""
    ### Key Engineering Highlights:
    1. **Dual Storage Layer**: Structured document metadata, chunk relations, and query telemetry are strictly partitioned in a SQL database with relational integrity, while dense normalized vectors are indexed for sub-millisecond similarity lookups.
    2. **Hybrid Retrieval (Dense + BM25)**: Overcomes dense embedding blind spots for technical identifiers (e.g. `P95`, `99.95%`, `SOC2`) by fusing dense semantic representations with Robertson-Spärck Jones BM25 via Reciprocal Rank Fusion ($k=60$).
    3. **Hallucination Prevention**: Prompt templates enforce strict citation grounding (`[Source: <doc> | Chunk: <id>]`) with automated fallback notices when relevance scores drop below safety thresholds.
    4. **Continuous MLOps Telemetry with MLflow**: Every query logs end-to-end execution traces, hyperparameters, retrieval latency, and evaluation metrics for experiment comparison and regression testing.
    """)
