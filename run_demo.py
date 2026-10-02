"""
Automated End-to-End Demo Script for GenAI Document RAG Assistant
Executes:
1. Document ingestion across PDF/TXT/CSV/MD sample enterprise files.
2. Structured SQL metadata & Dense Vector indexing.
3. Hybrid semantic retrieval (Dense + BM25 with RRF).
4. Grounded answer generation with source citations.
5. Metric evaluation (Relevance, MRR, Latency).
6. Automatic MLflow experiment tracking.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rag_assistant.pipeline import RAGPipeline
from rag_assistant.config import RAGConfig


def run_demo():
    print("=" * 70)
    print(" [*] INITIALIZING GENAI-POWERED DOCUMENT RAG ASSISTANT DEMO")
    print("=" * 70)

    config = RAGConfig()
    pipeline = RAGPipeline(config=config)

    # 1. Ingest Sample Documents
    sample_dir = Path("data/sample_docs")
    print(f"\n[1/3] Ingesting enterprise document repository from: {sample_dir}")
    results = pipeline.ingest_directory(sample_dir)
    print(f"      Ingested {len(results)} enterprise documents:")
    for r in results:
        print(f"      - {r['filename']:35} | Chunks: {r['total_chunks']:2} | Tokens: {r['total_tokens']:4}")

    # 2. Run Queries with Hybrid Retrieval
    demo_queries = [
        {
            "query": "What is the platform monthly uptime guarantee and what are the service credit tiers?",
            "mode": "hybrid",
        },
        {
            "query": "What are the security guidelines regarding external LLM API proxies and customer PII?",
            "mode": "hybrid",
        },
        {
            "query": "What are the latency SLAs for P95 API gateway and the disaster recovery RTO / RPO?",
            "mode": "hybrid",
        },
        {
            "query": "What was the allocated budget, actual spend, and ROI for the GenAI RAG Pipeline project?",
            "mode": "hybrid",
        },
    ]

    print(f"\n[2/3] Executing {len(demo_queries)} contextual queries with MLflow tracking...")
    print("-" * 70)

    for idx, item in enumerate(demo_queries, 1):
        q = item["query"]
        mode = item["mode"]
        print(f"\n--- [Query {idx}/{len(demo_queries)}] ---")
        print(f"Q: \"{q}\"")

        res = pipeline.query(query_text=q, retrieval_mode=mode, top_k=3, run_name=f"demo_run_{idx}")

        print("\nAnswer:")
        print(res["answer"])

        latency = res["latency"]
        eval_res = res["eval_result"]
        print(f"\nPerformance Telemetry:")
        print(f"  - Latency        : Retrieval={latency['retrieval_ms']}ms | Gen={latency['generation_ms']}ms | Total={latency['total_ms']}ms")
        print(f"  - Context Rel.   : {eval_res.context_relevance * 100:.1f}%")
        print(f"  - MRR Score      : {eval_res.mrr_score}")
        print(f"  - Avg Similarity : {eval_res.avg_similarity:.4f}")
        if res.get("mlflow_run_id"):
            print(f"  - MLflow Run ID  : {res['mlflow_run_id']}")
        print("-" * 70)

    # 3. Overall Stats
    print("\n[3/3] System Telemetry Summary (SQL & Vector DB):")
    stats = pipeline.get_stats()
    for k, v in stats.items():
        print(f"      {k:22}: {v}")

    print("\n" + "=" * 70)
    print(" [+] DEMO COMPLETED SUCCESSFULLY!")
    print(" Next steps:")
    print(" 1. Launch Streamlit UI : python -m streamlit run app.py")
    print(" 2. Launch MLflow UI    : python -m mlflow ui --backend-store-uri sqlite:///data/storage/mlflow.db")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
