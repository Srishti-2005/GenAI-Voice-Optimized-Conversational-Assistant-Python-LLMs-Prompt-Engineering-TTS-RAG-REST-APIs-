import argparse
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.config import RAGConfig


def main():
    parser = argparse.ArgumentParser(description="GenAI-Powered Document RAG Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into SQL and Vector DB")
    ingest_parser.add_argument("--path", type=str, default="data/sample_docs", help="File or directory path to ingest")

    # Command: Query
    query_parser = subparsers.add_parser("query", help="Ask a question against the document repository")
    query_parser.add_argument("query_text", type=str, help="Question to query")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    query_parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "dense", "bm25"], help="Retrieval mode")

    # Command: Stats
    subparsers.add_parser("stats", help="Display SQL and Vector DB statistics")

    # Command: Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-query benchmark and log to MLflow")
    bench_parser.add_argument("--runs", type=int, default=3, help="Number of test iterations")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    pipeline = RAGPipeline()

    if args.command == "ingest":
        p = Path(args.path)
        if p.is_dir():
            print(f"[*] Ingesting documents from directory: {p}")
            results = pipeline.ingest_directory(p)
            print(f"[+] Successfully processed {len(results)} files.")
            for r in results:
                print(f"    - {r['filename']}: {r['total_chunks']} chunks, {r['total_tokens']} tokens")
        elif p.is_file():
            print(f"[*] Ingesting single file: {p}")
            res = pipeline.ingest_file(p)
            print(f"[+] Done: {res['filename']} ({res['total_chunks']} chunks, {res['total_tokens']} tokens)")
        else:
            print(f"[!] Invalid path: {p}")

    elif args.command == "query":
        print(f"[*] Querying ({args.mode.upper()} mode, top_k={args.top_k}): '{args.query_text}'")
        res = pipeline.query(args.query_text, top_k=args.top_k, retrieval_mode=args.mode)
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(res["answer"])
        print("\n" + "-" * 60)
        print(f"Latencies: Retrieval={res['latency']['retrieval_ms']}ms | Generation={res['latency']['generation_ms']}ms | Total={res['latency']['total_ms']}ms")
        if res.get("mlflow_run_id"):
            print(f"MLflow Run ID: {res['mlflow_run_id']}")
        print(f"Evaluation: Relevance={res['eval_result'].context_relevance} | Avg Similarity={res['eval_result'].avg_similarity} | MRR={res['eval_result'].mrr_score}")
        print("-" * 60)

    elif args.command == "stats":
        stats = pipeline.get_stats()
        print("\n" + "=" * 50)
        print("ENTERPRISE RAG SYSTEM TELEMETRY (SQL & VECTOR)")
        print("=" * 50)
        for k, v in stats.items():
            print(f"  {k:25}: {v}")
        print("=" * 50)

    elif args.command == "benchmark":
        test_queries = [
            "What is the required platform uptime guarantee and credit calculation?",
            "What are the security guidelines for LLM and vector storage classification?",
            "What is the disaster recovery RTO and RPO in the cloud spec?",
            "What was the actual spend and ROI for the GenAI RAG Pipeline project in Q1-2025?",
        ]
        print(f"[*] Executing benchmark across {len(test_queries)} enterprise queries...")
        for i, q in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] Testing query: {q}")
            res = pipeline.query(q, retrieval_mode="hybrid", run_name=f"benchmark_query_{i}")
            print(f"    Latency: {res['latency']['total_ms']}ms | Relevance: {res['eval_result'].context_relevance} | MRR: {res['eval_result'].mrr_score} | MLflow Run: {res['mlflow_run_id']}")
        print("\n[+] Benchmark finished successfully. Results logged to MLflow and SQL telemetry.")


if __name__ == "__main__":
    main()
