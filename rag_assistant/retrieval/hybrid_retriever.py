import time
from typing import List, Dict, Any, Optional
import numpy as np
from rag_assistant.storage.vector_store import VectorStore
from rag_assistant.retrieval.bm25 import BM25Index
from rag_assistant.retrieval.embeddings import BaseEmbedder


class HybridRetriever:
    """
    Hybrid Retriever combining Dense Semantic Vectors with Sparse Lexical BM25
    using Reciprocal Rank Fusion (RRF) and normalized score blending.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = 60,
        default_alpha: float = 0.6,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k
        self.default_alpha = default_alpha  # Weight for dense (0.0 = pure bm25, 1.0 = pure dense)

    def update_bm25_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Fit BM25 on the latest document chunks."""
        self.bm25_index.fit(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "hybrid",
        alpha: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        mode = mode.lower()
        effective_alpha = self.default_alpha if alpha is None else alpha

        # 1. Dense Semantic Retrieval
        dense_results: List[Dict[str, Any]] = []
        if mode in ("dense", "hybrid"):
            query_vector = self.embedder.embed_query(query)
            dense_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # Fetch pool for fusion
                filter_dict=filter_dict,
            )

        # 2. Sparse Lexical BM25 Retrieval
        bm25_results: List[Dict[str, Any]] = []
        if mode in ("bm25", "hybrid"):
            bm25_results = self.bm25_index.search(query=query, top_k=top_k * 2)

        # 3. Combine / Route
        if mode == "dense":
            final_chunks = dense_results[:top_k]
            for c in final_chunks:
                c["final_score"] = c.get("similarity_score", 0.0)

        elif mode == "bm25":
            final_chunks = bm25_results[:top_k]
            for c in final_chunks:
                c["final_score"] = c.get("bm25_score", 0.0)

        else:
            # Hybrid Fusion (Reciprocal Rank Fusion + Normalized Score Blending)
            final_chunks = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                alpha=effective_alpha,
                top_k=top_k,
            )

        retrieval_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunks": final_chunks,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "dense_count": len(dense_results),
            "bm25_count": len(bm25_results),
        }

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        alpha: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scores_map: Dict[str, Dict[str, Any]] = {}

        # Dense contribution
        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            rrf_contrib = alpha * (1.0 / (self.rrf_k + (rank + 1)))
            if cid not in scores_map:
                scores_map[cid] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "dense_score": item.get("similarity_score", 0.0),
                    "bm25_score": 0.0,
                    "dense_rank": rank + 1,
                    "bm25_rank": None,
                }
            scores_map[cid]["rrf_score"] += rrf_contrib
            scores_map[cid]["dense_score"] = item.get("similarity_score", 0.0)

        # BM25 contribution
        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_contrib = (1.0 - alpha) * (1.0 / (self.rrf_k + (rank + 1)))
            if cid not in scores_map:
                scores_map[cid] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "dense_score": 0.0,
                    "bm25_score": item.get("bm25_score", 0.0),
                    "dense_rank": None,
                    "bm25_rank": rank + 1,
                }
            scores_map[cid]["rrf_score"] += rrf_contrib
            scores_map[cid]["bm25_score"] = item.get("bm25_score", 0.0)
            scores_map[cid]["bm25_rank"] = rank + 1

        # Sort by total fused score
        sorted_items = sorted(scores_map.values(), key=lambda x: x["rrf_score"], reverse=True)

        results = []
        for idx, entry in enumerate(sorted_items[:top_k]):
            chunk = dict(entry["chunk"])
            chunk["final_score"] = float(entry["rrf_score"])
            chunk["dense_score"] = float(entry["dense_score"])
            chunk["bm25_score"] = float(entry["bm25_score"])
            chunk["dense_rank"] = entry["dense_rank"]
            chunk["bm25_rank"] = entry["bm25_rank"]
            chunk["hybrid_rank"] = idx + 1
            results.append(chunk)

        return results
