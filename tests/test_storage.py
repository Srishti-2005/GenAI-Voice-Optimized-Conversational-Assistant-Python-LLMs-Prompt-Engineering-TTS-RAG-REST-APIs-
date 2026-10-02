import unittest
import tempfile
from pathlib import Path
from rag_assistant.storage.sql_store import SQLDocumentStore
from rag_assistant.storage.vector_store import VectorStore
from rag_assistant.ingestion.parser import ParsedDocument
from rag_assistant.ingestion.chunker import DocumentChunk
import numpy as np


class TestStorage(unittest.TestCase):
    def test_sql_document_store(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            db_file = Path(tmp_dir) / "test.db"
            store = SQLDocumentStore(db_file)

            doc = ParsedDocument(
                doc_id="test_doc_1",
                filename="policy.txt",
                file_type=".txt",
                file_size_bytes=1024,
                content="Sample policy content",
            )
            store.insert_document(doc, total_chunks=1, total_tokens=10)

            docs = store.get_all_documents()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["filename"], "policy.txt")

            chunk = DocumentChunk(
                chunk_id="test_doc_1_c001",
                doc_id="test_doc_1",
                filename="policy.txt",
                chunk_index=0,
                content="Sample policy content",
                token_count=10,
            )
            store.insert_chunks([chunk])
            chunks = store.get_chunks_by_doc_id("test_doc_1")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["chunk_id"], "test_doc_1_c001")

    def test_vector_store(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            idx_file = Path(tmp_dir) / "index.npz"
            meta_file = Path(tmp_dir) / "meta.json"
            vstore = VectorStore(idx_file, meta_file)

            chunks = [
                DocumentChunk(chunk_id="c1", doc_id="d1", filename="f1.txt", chunk_index=0, content="cat", token_count=1),
                DocumentChunk(chunk_id="c2", doc_id="d1", filename="f1.txt", chunk_index=1, content="dog", token_count=1),
            ]
            matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            vstore.add_chunks(chunks, matrix)

            self.assertEqual(len(vstore), 2)
            query_vec = np.array([1.0, 0.0], dtype=np.float32)
            results = vstore.search(query_vec, top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], "c1")
            self.assertGreater(results[0]["similarity_score"], 0.99)


if __name__ == "__main__":
    unittest.main()
