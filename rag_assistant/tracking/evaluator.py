import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EvaluationResult:
    avg_similarity: float
    context_relevance: float
    mrr_score: float
    precision_at_k: float
    citation_coverage: float
    details: Dict[str, Any] = field(default_factory=dict)


class RAGEvaluator:
    """Evaluates retrieval quality, ranking efficacy, and answer grounding."""

    def __init__(self, relevance_threshold: float = 0.25):
        self.relevance_threshold = relevance_threshold

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        response_text: str,
        k: int = 4,
    ) -> EvaluationResult:
        if not retrieved_chunks:
            return EvaluationResult(
                avg_similarity=0.0,
                context_relevance=0.0,
                mrr_score=0.0,
                precision_at_k=0.0,
                citation_coverage=0.0,
                details={"reason": "No chunks retrieved"},
            )

        query_terms = set(re.findall(r"[a-zA-Z0-9_\-\.\%]+", query.lower()))
        sim_scores = [c.get("similarity_score") or c.get("final_score") or 0.0 for c in retrieved_chunks]
        avg_sim = float(sum(sim_scores) / len(sim_scores)) if sim_scores else 0.0

        # Precision@K & MRR
        relevant_flags = []
        for c in retrieved_chunks:
            chunk_text = c.get("content", "").lower()
            overlap = sum(1 for term in query_terms if term in chunk_text)
            # Relevant if it overlaps with query keywords or has high vector similarity
            is_rel = overlap > 0 or (c.get("similarity_score", 0.0) >= self.relevance_threshold)
            relevant_flags.append(is_rel)

        precision_at_k = sum(1 for r in relevant_flags[:k] if r) / float(min(k, len(retrieved_chunks)))

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                mrr = 1.0 / rank
                break

        # Context Relevance: token recall in retrieved contexts
        all_context_text = " ".join([c.get("content", "").lower() for c in retrieved_chunks])
        if query_terms:
            matched_terms = sum(1 for term in query_terms if term in all_context_text)
            context_relevance = matched_terms / len(query_terms)
        else:
            context_relevance = 1.0

        # Citation Coverage: checks if response contains cited source filenames or chunk IDs
        cited_count = 0
        for c in retrieved_chunks:
            cid = c.get("chunk_id", "")
            fname = c.get("filename", "")
            if (cid and cid in response_text) or (fname and fname in response_text):
                cited_count += 1
        citation_coverage = cited_count / len(retrieved_chunks) if retrieved_chunks else 0.0

        return EvaluationResult(
            avg_similarity=round(avg_sim, 4),
            context_relevance=round(context_relevance, 4),
            mrr_score=round(mrr, 4),
            precision_at_k=round(precision_at_k, 4),
            citation_coverage=round(citation_coverage, 4),
            details={
                "retrieved_count": len(retrieved_chunks),
                "relevant_count": sum(1 for r in relevant_flags if r),
                "query_terms_count": len(query_terms),
            },
        )
