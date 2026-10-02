import unittest
from rag_assistant.retrieval.bm25 import BM25Index
from rag_assistant.retrieval.embeddings import LocalSemanticEmbedder
import numpy as np


class TestRetrieval(unittest.TestCase):
    def test_bm25_retrieval(self):
        index = BM25Index()
        chunks = [
            {"chunk_id": "c1", "content": "AWS Aurora PostgreSQL database with replication"},
            {"chunk_id": "c2", "content": "Machine learning model deployment with Kubernetes"},
            {"chunk_id": "c3", "content": "Customer support service level agreement 99.95% uptime"},
        ]
        index.fit(chunks)

        res = index.search("Aurora database replication", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["chunk_id"], "c1")

        res_sla = index.search("uptime agreement", top_k=1)
        self.assertEqual(len(res_sla), 1)
        self.assertEqual(res_sla[0]["chunk_id"], "c3")

    def test_local_embedder(self):
        embedder = LocalSemanticEmbedder(dimension=384)
        texts = [
            "Kubernetes orchestration in us-east-1",
            "Disaster recovery and failover objective RTO",
        ]
        embeddings = embedder.embed_texts(texts)
        self.assertEqual(embeddings.shape, (2, 384))
        norms = np.linalg.norm(embeddings, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
