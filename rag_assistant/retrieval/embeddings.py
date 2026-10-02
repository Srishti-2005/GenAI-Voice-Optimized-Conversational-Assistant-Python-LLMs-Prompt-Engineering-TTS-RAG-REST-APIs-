import os
import re
import json
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import requests


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Takes a list of texts and returns a 2D float32 numpy array of shape (N, D)."""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D vector."""
        return self.embed_texts([query])[0]


class LocalSemanticEmbedder(BaseEmbedder):
    """
    High-performance, zero-external-dependency local semantic embedder.
    Combines character n-gram subword hashing, term-frequency weighting,
    and pseudo-random orthogonal projection into a dense 384-dimensional vector space.
    Guarantees reproducible, normalized cosine semantic representations
    for technical, enterprise, and domain-specific vocabulary offline.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Deterministic projection seed
        rng = np.random.RandomState(42)
        self.projection_matrix = rng.normal(loc=0.0, scale=1.0 / np.sqrt(dimension), size=(4096, dimension)).astype(np.float32)

    def _hash_token(self, token: str) -> int:
        h = 2166136261
        for char in token:
            h = (h ^ ord(char)) * 16777619
            h &= 0xFFFFFFFF
        return h % 4096

    def _extract_features(self, text: str) -> np.ndarray:
        # Tokenize words and subwords (n-grams of length 3-5)
        text_clean = text.lower()
        tokens = re.findall(r"[a-z0-9_\-\.\%]+", text_clean)

        feat_vector = np.zeros(4096, dtype=np.float32)

        for token in tokens:
            idx = self._hash_token(token)
            feat_vector[idx] += 1.5  # Full token weight

            # Subword character n-grams for typo resilience and morphology
            if len(token) > 3:
                for n in range(3, min(6, len(token))):
                    for i in range(len(token) - n + 1):
                        sub = token[i : i + n]
                        sub_idx = self._hash_token(sub)
                        feat_vector[sub_idx] += 0.4

        # BM25-style sublinear scaling: log(1 + tf)
        feat_vector = np.log1p(feat_vector)
        return feat_vector

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            feats = self._extract_features(text)
            # Project 4096 sparse features to dense 384 dimensions
            dense_vec = np.dot(feats, self.projection_matrix)

            # L2 normalization for cosine similarity
            norm = np.linalg.norm(dense_vec)
            if norm > 0:
                dense_vec = dense_vec / norm
            else:
                dense_vec = np.zeros(self.dimension, dtype=np.float32)

            embeddings.append(dense_vec)

        return np.array(embeddings, dtype=np.float32)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API Embedding client (text-embedding-3-small)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.openai.com/v1/embeddings"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Batch in chunks of 50
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"input": batch, "model": self.model}
            resp = requests.post(self.url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI Embedding API Error ({resp.status_code}): {resp.text}")
            data = resp.json()
            sorted_items = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend([item["embedding"] for item in sorted_items])

        return np.array(all_embeddings, dtype=np.float32)


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini Embedding client (text-embedding-004)."""

    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={self.api_key}"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        requests_payload = [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}} for t in texts]
        resp = requests.post(self.url, json={"requests": requests_payload}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini Embedding API Error ({resp.status_code}): {resp.text}")
        data = resp.json()
        embeddings = [item["values"] for item in data.get("embeddings", [])]
        return np.array(embeddings, dtype=np.float32)


class OllamaEmbedder(BaseEmbedder):
    """Local Ollama Embedding client (e.g. nomic-embed-text)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API Error: {resp.text}")
            data = resp.json()
            embeddings.append(data["embedding"])
        return np.array(embeddings, dtype=np.float32)


class EmbeddingFactory:
    """Factory to instantiate embedding providers with automatic fallback."""

    @staticmethod
    def create(
        provider: str = "local",
        openai_api_key: str = "",
        gemini_api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
    ) -> BaseEmbedder:
        provider = provider.lower()

        if provider == "openai" and openai_api_key:
            return OpenAIEmbedder(api_key=openai_api_key)
        elif provider == "gemini" and gemini_api_key:
            return GeminiEmbedder(api_key=gemini_api_key)
        elif provider == "ollama":
            return OllamaEmbedder(base_url=ollama_base_url)
        else:
            # Default robust offline semantic embedder
            return LocalSemanticEmbedder()
