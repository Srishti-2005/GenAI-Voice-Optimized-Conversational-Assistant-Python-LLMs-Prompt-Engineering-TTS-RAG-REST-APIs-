import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from rag_assistant.ingestion.parser import ParsedDocument
from rag_assistant.ingestion.chunker import DocumentChunk


class SQLDocumentStore:
    """
    SQL relational storage for enterprise document catalog,
    chunk relationships, telemetry logs, and evaluation metrics.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Documents catalog table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'indexed',
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Document Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                );
            """)

            # Index for fast relational lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_filename ON document_chunks(filename);")

            # 3. Query telemetry and interaction audit table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    query_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    response_text TEXT,
                    retrieval_mode TEXT,
                    top_k INTEGER,
                    retrieval_latency_ms REAL,
                    generation_latency_ms REAL,
                    total_latency_ms REAL,
                    retrieved_chunks_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. Retrieval evaluation metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_metrics (
                    eval_id TEXT PRIMARY KEY,
                    query_id TEXT,
                    experiment_name TEXT,
                    avg_similarity REAL,
                    context_relevance REAL,
                    mrr_score REAL,
                    precision_at_k REAL,
                    eval_summary_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES query_logs (query_id) ON DELETE SET NULL
                );
            """)

            conn.commit()

    def insert_document(self, doc: ParsedDocument, total_chunks: int, total_tokens: int) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (doc_id, filename, file_type, file_size_bytes, total_chunks, total_tokens, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    file_size_bytes=excluded.file_size_bytes,
                    total_chunks=excluded.total_chunks,
                    total_tokens=excluded.total_tokens,
                    metadata_json=excluded.metadata_json;
                """,
                (
                    doc.doc_id,
                    doc.filename,
                    doc.file_type,
                    doc.file_size_bytes,
                    total_chunks,
                    total_tokens,
                    json.dumps(doc.metadata),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def insert_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            records = [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.chunk_index,
                    c.filename,
                    c.content,
                    c.token_count,
                    json.dumps(c.metadata),
                    datetime.now(timezone.utc).isoformat(),
                )
                for c in chunks
            ]
            cursor.executemany(
                """
                INSERT INTO document_chunks (chunk_id, doc_id, chunk_index, filename, content, token_count, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    content=excluded.content,
                    token_count=excluded.token_count,
                    metadata_json=excluded.metadata_json;
                """,
                records,
            )
            conn.commit()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?;", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index ASC;", (doc_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM document_chunks ORDER BY doc_id, chunk_index ASC;")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def log_query(
        self,
        query_id: str,
        query_text: str,
        response_text: str,
        retrieval_mode: str,
        top_k: int,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_logs (
                    query_id, query_text, response_text, retrieval_mode, top_k,
                    retrieval_latency_ms, generation_latency_ms, total_latency_ms,
                    retrieved_chunks_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    query_id,
                    query_text,
                    response_text,
                    retrieval_mode,
                    top_k,
                    retrieval_latency_ms,
                    generation_latency_ms,
                    total_latency_ms,
                    json.dumps(retrieved_chunks),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def log_eval_metrics(
        self,
        eval_id: str,
        query_id: str,
        experiment_name: str,
        avg_similarity: float,
        context_relevance: float,
        mrr_score: float,
        precision_at_k: float,
        summary: Dict[str, Any],
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO eval_metrics (
                    eval_id, query_id, experiment_name, avg_similarity,
                    context_relevance, mrr_score, precision_at_k,
                    eval_summary_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    eval_id,
                    query_id,
                    experiment_name,
                    avg_similarity,
                    context_relevance,
                    mrr_score,
                    precision_at_k,
                    json.dumps(summary),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_system_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total_docs, COALESCE(SUM(total_chunks), 0) AS total_chunks, COALESCE(SUM(total_tokens), 0) AS total_tokens FROM documents;")
            doc_stats = dict(cursor.fetchone() or {})

            cursor.execute("SELECT COUNT(*) AS total_queries, COALESCE(AVG(total_latency_ms), 0) AS avg_latency_ms FROM query_logs;")
            query_stats = dict(cursor.fetchone() or {})

            cursor.execute("SELECT COALESCE(AVG(avg_similarity), 0) AS avg_relevance, COALESCE(AVG(mrr_score), 0) AS avg_mrr FROM eval_metrics;")
            eval_stats = dict(cursor.fetchone() or {})

            return {**doc_stats, **query_stats, **eval_stats}
