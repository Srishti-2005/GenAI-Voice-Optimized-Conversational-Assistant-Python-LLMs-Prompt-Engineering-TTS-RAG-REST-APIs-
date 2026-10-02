import unittest
import tempfile
from pathlib import Path
from rag_assistant.config import RAGConfig
from rag_assistant.pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    def test_full_rag_pipeline_flow(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = RAGConfig(
                storage_dir=tmp_path / "storage",
                db_path=tmp_path / "storage" / "docs.db",
                vector_index_path=tmp_path / "storage" / "vec.npz",
                metadata_store_path=tmp_path / "storage" / "meta.json",
                mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'storage' / 'mlflow.db').as_posix()}",
                chunk_size=300,
                chunk_overlap=50,
                llm_provider="offline",
            )

            pipeline = RAGPipeline(config=cfg)

            # Ingest sample policy
            sample_doc = Path("data/sample_docs/enterprise_ai_policy.txt")
            res = pipeline.ingest_file(sample_doc)
            self.assertEqual(res["status"], "success")
            self.assertGreater(res["total_chunks"], 0)

            # Query
            query_text = "What is the policy for external LLM API calls and PII?"
            answer_data = pipeline.query(query_text, top_k=2, retrieval_mode="hybrid")

            self.assertIn("answer", answer_data)
            self.assertGreater(len(answer_data["answer"]), 20)
            self.assertGreater(len(answer_data["retrieved_chunks"]), 0)
            self.assertIsNotNone(answer_data["eval_result"])
            self.assertGreater(answer_data["eval_result"].context_relevance, 0.0)


if __name__ == "__main__":
    unittest.main()
