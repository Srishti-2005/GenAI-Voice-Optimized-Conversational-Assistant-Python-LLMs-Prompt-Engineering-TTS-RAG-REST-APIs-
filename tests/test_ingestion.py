import unittest
from pathlib import Path
from rag_assistant.ingestion.parser import DocumentParser
from rag_assistant.ingestion.chunker import DocumentChunker


class TestIngestion(unittest.TestCase):
    def test_parser_sample_files(self):
        parser = DocumentParser()
        sample_dir = Path("data/sample_docs")

        txt_file = sample_dir / "enterprise_ai_policy.txt"
        self.assertTrue(txt_file.exists())
        doc_txt = parser.parse_file(txt_file)
        self.assertEqual(doc_txt.filename, "enterprise_ai_policy.txt")
        self.assertGreater(len(doc_txt.content), 100)
        self.assertIn("Governance", doc_txt.content)

        csv_file = sample_dir / "quarterly_financial_report.csv"
        self.assertTrue(csv_file.exists())
        doc_csv = parser.parse_file(csv_file)
        self.assertEqual(doc_csv.file_type, ".csv")
        self.assertIn("Q1-2025", doc_csv.content)

    def test_chunker_sliding_window(self):
        parser = DocumentParser()
        sample_file = Path("data/sample_docs/enterprise_ai_policy.txt")
        doc = parser.parse_file(sample_file)

        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertGreater(len(chunk.content), 0)
            self.assertGreater(chunk.token_count, 0)
            self.assertEqual(chunk.doc_id, doc.doc_id)


if __name__ == "__main__":
    unittest.main()
