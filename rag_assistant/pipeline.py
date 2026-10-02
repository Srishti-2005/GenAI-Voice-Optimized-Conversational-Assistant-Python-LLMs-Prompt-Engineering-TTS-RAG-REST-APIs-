import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag_assistant.config import RAGConfig
from rag_assistant.ingestion.parser import DocumentParser, ParsedDocument
from rag_assistant.ingestion.chunker import DocumentChunker, DocumentChunk
from rag_assistant.storage.sql_store import SQLDocumentStore
from rag_assistant.storage.vector_store import VectorStore
from rag_assistant.retrieval.embeddings import EmbeddingFactory, BaseEmbedder
from rag_assistant.retrieval.bm25 import BM25Index
from rag_assistant.retrieval.hybrid_retriever import HybridRetriever
from rag_assistant.generation.llm_client import LLMFactory, BaseLLMClient
from rag_assistant.tracking.evaluator import RAGEvaluator, EvaluationResult
from rag_assistant.tracking.mlflow_tracker import MLflowTracker


class RAGPipeline:
    """
    End-to-End Enterprise RAG Pipeline orchestrating:
    - Multi-format Ingestion & Semantic Chunking
    - Relational SQL Catalog & Dense Vector Storage
    - Hybrid Retrieval (Dense Vector + BM25 Lexical + Reciprocal Rank Fusion)
    - Grounded LLM Generation with Citation Tracking
    - MLflow Experiment & Retrieval Metric Tracking
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # 1. Initialize Ingestion Components
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        # 2. Initialize Storage Layers
        self.sql_store = SQLDocumentStore(db_path=self.config.db_path)
        self.vector_store = VectorStore(
            index_path=self.config.vector_index_path,
            metadata_path=self.config.metadata_store_path,
        )

        # 3. Initialize Embeddings & BM25
        self.embedder: BaseEmbedder = EmbeddingFactory.create(
            provider=self.config.embedding_provider,
            openai_api_key=self.config.openai_api_key,
            gemini_api_key=self.config.gemini_api_key,
            ollama_base_url=self.config.ollama_base_url,
        )
        self.bm25_index = BM25Index()

        # 4. Initialize Hybrid Retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            bm25_index=self.bm25_index,
            default_alpha=self.config.hybrid_alpha,
        )

        # Sync BM25 with existing storage chunks if already populated
        existing_chunks = self.sql_store.get_all_chunks()
        if existing_chunks:
            self.retriever.update_bm25_index(existing_chunks)

        # 5. Initialize LLM Client
        self.llm_client: BaseLLMClient = LLMFactory.create(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            openai_api_key=self.config.openai_api_key,
            gemini_api_key=self.config.gemini_api_key,
            ollama_base_url=self.config.ollama_base_url,
        )

        # 6. Initialize Tracking & Evaluation
        self.evaluator = RAGEvaluator(relevance_threshold=self.config.min_similarity_threshold)
        self.mlflow_tracker = None
        if self.config.enable_mlflow:
            self.mlflow_tracker = MLflowTracker(
                tracking_uri=self.config.mlflow_tracking_uri,
                experiment_name=self.config.mlflow_experiment_name,
            )

    def ingest_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Ingest a single document file from disk."""
        parsed_doc = self.parser.parse_file(file_path)
        return self._process_and_store_document(parsed_doc)

    def ingest_bytes(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Ingest a document from memory/file upload."""
        parsed_doc = self.parser.parse_bytes(filename, file_bytes)
        return self._process_and_store_document(parsed_doc)

    def ingest_directory(self, dir_path: str | Path) -> List[Dict[str, Any]]:
        """Ingest all supported files from a directory."""
        dir_p = Path(dir_path)
        results = []
        for ext in DocumentParser.SUPPORTED_EXTENSIONS:
            for file_path in dir_p.glob(f"*{ext}"):
                res = self.ingest_file(file_path)
                results.append(res)
        return results

    def _process_and_store_document(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        # 1. Chunk document
        chunks: List[DocumentChunk] = self.chunker.chunk_document(parsed_doc)
        total_tokens = sum(c.token_count for c in chunks)

        # 2. Embed chunks
        if chunks:
            chunk_texts = [c.content for c in chunks]
            embeddings_matrix = self.embedder.embed_texts(chunk_texts)
            # Store in Dense Vector Store
            self.vector_store.add_chunks(chunks, embeddings_matrix)

        # 3. Store in SQL Relational Store
        self.sql_store.insert_document(parsed_doc, total_chunks=len(chunks), total_tokens=total_tokens)
        self.sql_store.insert_chunks(chunks)

        # 4. Update BM25 Index
        all_chunks = self.sql_store.get_all_chunks()
        self.retriever.update_bm25_index(all_chunks)

        return {
            "doc_id": parsed_doc.doc_id,
            "filename": parsed_doc.filename,
            "file_type": parsed_doc.file_type,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "status": "success",
        }

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full RAG query pipeline:
        Retrieval -> LLM Generation -> Evaluation -> SQL Audit Logging -> MLflow Run Tracking
        """
        total_start = time.perf_counter()
        query_id = f"qry_{uuid.uuid4().hex[:12]}"
        effective_top_k = top_k or self.config.top_k
        effective_mode = retrieval_mode or self.config.retrieval_mode
        effective_temp = temperature if temperature is not None else self.config.temperature

        # 1. Retrieve Context Chunks
        retrieval_res = self.retriever.retrieve(
            query=query_text,
            top_k=effective_top_k,
            mode=effective_mode,
            alpha=alpha,
        )
        retrieved_chunks = retrieval_res["chunks"]
        retrieval_latency = retrieval_res["retrieval_latency_ms"]

        # 2. Generate LLM Answer
        gen_res = self.llm_client.generate(
            query=query_text,
            context_chunks=retrieved_chunks,
            temperature=effective_temp,
        )
        response_text = gen_res["answer"]
        generation_latency = gen_res["generation_latency_ms"]
        tokens_used = gen_res.get("tokens_used", 0)

        total_latency_ms = round((time.perf_counter() - total_start) * 1000.0, 2)

        # 3. Evaluate Retrieval Quality
        eval_result = self.evaluator.evaluate(
            query=query_text,
            retrieved_chunks=retrieved_chunks,
            response_text=response_text,
            k=effective_top_k,
        )

        # 4. Log to SQL Database
        self.sql_store.log_query(
            query_id=query_id,
            query_text=query_text,
            response_text=response_text,
            retrieval_mode=effective_mode,
            top_k=effective_top_k,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            total_latency_ms=total_latency_ms,
            retrieved_chunks=retrieved_chunks,
        )
        self.sql_store.log_eval_metrics(
            eval_id=f"eval_{query_id}",
            query_id=query_id,
            experiment_name=self.config.mlflow_experiment_name,
            avg_similarity=eval_result.avg_similarity,
            context_relevance=eval_result.context_relevance,
            mrr_score=eval_result.mrr_score,
            precision_at_k=eval_result.precision_at_k,
            summary=eval_result.details,
        )

        # 5. Log to MLflow
        mlflow_run_id = None
        if self.mlflow_tracker:
            parameters = {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k": effective_top_k,
                "retrieval_mode": effective_mode,
                "hybrid_alpha": alpha if alpha is not None else self.config.hybrid_alpha,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "temperature": effective_temp,
                "embedding_provider": self.config.embedding_provider,
            }
            metrics = {
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": generation_latency,
                "total_latency_ms": total_latency_ms,
                "retrieved_chunks_count": len(retrieved_chunks),
                "tokens_used": tokens_used,
            }
            mlflow_run_id = self.mlflow_tracker.log_rag_run(
                run_name=run_name or f"rag_{effective_mode}_{effective_top_k}",
                parameters=parameters,
                metrics=metrics,
                query=query_text,
                response_text=response_text,
                retrieved_chunks=retrieved_chunks,
                eval_result=eval_result,
            )

        return {
            "query_id": query_id,
            "query": query_text,
            "answer": response_text,
            "retrieved_chunks": retrieved_chunks,
            "eval_result": eval_result,
            "latency": {
                "retrieval_ms": retrieval_latency,
                "generation_ms": generation_latency,
                "total_ms": total_latency_ms,
            },
            "parameters": {
                "mode": effective_mode,
                "top_k": effective_top_k,
                "llm_model": self.config.llm_model,
            },
            "mlflow_run_id": mlflow_run_id,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate system statistics from SQL and Vector index."""
        sql_stats = self.sql_store.get_system_stats()
        sql_stats["vector_index_size"] = len(self.vector_store)
        return sql_stats
