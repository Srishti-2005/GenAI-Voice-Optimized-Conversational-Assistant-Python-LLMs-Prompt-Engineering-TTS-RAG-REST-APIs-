import math
import re
from typing import List, Dict, Any, Tuple


class BM25Index:
    """
    Okapi BM25 implementation for lexical keyword search.
    Provides term-exact matching to complement dense semantic embeddings.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Lowers, extracts alphanumeric tokens and technical identifiers (e.g. p95, 99.95%, v3.2)
        tokens = re.findall(r"[a-zA-Z0-9_\-\.\%]+", text.lower())
        # Filter purely 1-char punctuation artifacts
        return [t for t in tokens if len(t) > 1 or t.isalnum()]

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """Index the provided list of chunk dicts."""
        self.corpus_chunks = chunks
        self.doc_ids = [c["chunk_id"] for c in chunks]
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        N = len(chunks)
        if N == 0:
            self.avg_doc_len = 0.0
            return

        for chunk in chunks:
            tokens = self.tokenize(chunk["content"])
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_term_freqs.append(freqs)

            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / N if N > 0 else 0.0

        # Precompute Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Compute BM25 scores for query across all indexed chunks."""
        if not self.corpus_chunks or self.avg_doc_len == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * len(self.corpus_chunks)

        for i, term_freqs in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lengths[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)

            score = 0.0
            for token in query_tokens:
                if token in term_freqs:
                    tf = term_freqs[token]
                    idf = self.idf.get(token, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * len_norm
                    score += idf * (numerator / denominator)

            scores[i] = score

        # Rank documents
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            if scores[idx] <= 0:
                continue
            chunk_data = dict(self.corpus_chunks[idx])
            chunk_data["bm25_score"] = float(scores[idx])
            chunk_data["bm25_rank"] = rank + 1
            results.append(chunk_data)

        return results
