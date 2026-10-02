import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rag_assistant.ingestion.chunker import DocumentChunk


class VectorStore:
    """
    Vector Store for dense semantic embeddings and cosine similarity search.
    Features:
    - In-memory normalized embedding matrix with sub-millisecond dot product
    - Persistent disk serialization (.npz + .json)
    - Metadata filtering and score thresholding
    """

    def __init__(self, index_path: str | Path, metadata_path: str | Path):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, D)
        self.chunks_metadata: List[Dict[str, Any]] = []  # Length: N
        self.chunk_id_to_index: Dict[str, int] = {}

        self.load()

    def __len__(self) -> int:
        return len(self.chunks_metadata)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings_matrix: np.ndarray) -> None:
        """
        Add or update chunks with their dense embeddings.
        embeddings_matrix should have shape (len(chunks), D).
        """
        if len(chunks) == 0:
            return

        if len(chunks) != len(embeddings_matrix):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match embeddings rows ({len(embeddings_matrix)})")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_embeddings = embeddings_matrix / norms

        if self.embeddings is None or len(self.chunks_metadata) == 0:
            self.embeddings = norm_embeddings
            self.chunks_metadata = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "token_count": c.token_count,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
        else:
            # Append or update existing
            existing_ids = {item["chunk_id"]: idx for idx, item in enumerate(self.chunks_metadata)}
            new_embeddings_list = list(self.embeddings)

            for i, chunk in enumerate(chunks):
                if chunk.chunk_id in existing_ids:
                    # Update
                    idx = existing_ids[chunk.chunk_id]
                    self.chunks_metadata[idx] = {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "filename": chunk.filename,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "metadata": chunk.metadata,
                    }
                    new_embeddings_list[idx] = norm_embeddings[i]
                else:
                    self.chunks_metadata.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "filename": chunk.filename,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "token_count": chunk.token_count,
                            "metadata": chunk.metadata,
                        }
                    )
                    new_embeddings_list.append(norm_embeddings[i])

            self.embeddings = np.array(new_embeddings_list, dtype=np.float32)

        self._reindex_id_map()
        self.save()

    def _reindex_id_map(self):
        self.chunk_id_to_index = {item["chunk_id"]: i for i, item in enumerate(self.chunks_metadata)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cosine similarity search over normalized embeddings.
        Returns sorted list of chunk dicts with 'similarity_score'.
        """
        if self.embeddings is None or len(self.chunks_metadata) == 0:
            return []

        # Ensure query is 1D and normalized
        q_vec = query_vector.flatten()
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Dot product gives cosine similarity
        scores = np.dot(self.embeddings, q_vec)

        # Apply metadata filter if provided
        valid_indices = []
        for idx, item in enumerate(self.chunks_metadata):
            if filter_dict:
                matches = all(item.get(k) == v or item.get("metadata", {}).get(k) == v for k, v in filter_dict.items())
                if not matches:
                    continue
            if scores[idx] >= score_threshold:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        # Sort top-k
        sorted_indices = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(sorted_indices):
            chunk_data = dict(self.chunks_metadata[idx])
            chunk_data["similarity_score"] = float(scores[idx])
            chunk_data["dense_rank"] = rank + 1
            results.append(chunk_data)

        return results

    def save(self) -> None:
        """Persist embeddings matrix and metadata to disk."""
        if self.embeddings is not None and len(self.chunks_metadata) > 0:
            np.savez_compressed(self.index_path, embeddings=self.embeddings)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_metadata, f, indent=2)

    def load(self) -> None:
        """Load embeddings and metadata from disk if available."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                data = np.load(self.index_path)
                self.embeddings = data["embeddings"]
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.chunks_metadata = json.load(f)
                self._reindex_id_map()
            except Exception as e:
                # Corrupted or outdated index
                self.embeddings = None
                self.chunks_metadata = []
                self.chunk_id_to_index = {}
