import os
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
from pypdf import PdfReader


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[str] = field(default_factory=list)


class DocumentParser:
    """Multi-format enterprise document parser supporting PDF, TXT, MD, CSV, and JSON."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()[:16]

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file does not exist: {path}")

        file_type = path.suffix.lower()
        if file_type not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document format '{file_type}'. Supported: {self.SUPPORTED_EXTENSIONS}")

        file_bytes = path.read_bytes()
        file_size = len(file_bytes)
        doc_id = f"doc_{self.compute_file_hash(file_bytes)}"

        if file_type == ".pdf":
            return self._parse_pdf(path, doc_id, file_size)
        elif file_type in {".txt", ".md"}:
            return self._parse_text(path, doc_id, file_size, file_type)
        elif file_type == ".csv":
            return self._parse_csv(path, doc_id, file_size)
        elif file_type == ".json":
            return self._parse_json(path, doc_id, file_size)
        else:
            raise ValueError(f"Unhandled file extension: {file_type}")

    def parse_bytes(self, filename: str, file_bytes: bytes) -> ParsedDocument:
        file_type = Path(filename).suffix.lower()
        if file_type not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{file_type}'.")

        file_size = len(file_bytes)
        doc_id = f"doc_{self.compute_file_hash(file_bytes)}"

        if file_type == ".pdf":
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
            full_text = "\n\n".join(pages_text)
            return ParsedDocument(
                doc_id=doc_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size,
                content=full_text,
                metadata={"total_pages": len(reader.pages), "word_count": len(full_text.split())},
                pages=pages_text,
            )
        elif file_type in {".txt", ".md"}:
            text = file_bytes.decode("utf-8", errors="replace")
            return ParsedDocument(
                doc_id=doc_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size,
                content=text,
                metadata={"word_count": len(text.split()), "lines_count": len(text.splitlines())},
                pages=[text],
            )
        elif file_type == ".csv":
            import io
            df = pd.read_csv(io.BytesIO(file_bytes))
            return self._df_to_parsed_doc(df, filename, doc_id, file_size)
        elif file_type == ".json":
            data = json.loads(file_bytes.decode("utf-8", errors="replace"))
            formatted = json.dumps(data, indent=2)
            return ParsedDocument(
                doc_id=doc_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size,
                content=formatted,
                metadata={"keys": list(data.keys()) if isinstance(data, dict) else len(data)},
                pages=[formatted],
            )
        raise ValueError(f"Unhandled file extension: {file_type}")

    def _parse_pdf(self, path: Path, doc_id: str, file_size: int) -> ParsedDocument:
        reader = PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type=".pdf",
            file_size_bytes=file_size,
            content=full_text,
            metadata={
                "total_pages": len(reader.pages),
                "word_count": len(full_text.split()),
                "author": reader.metadata.author if reader.metadata else None,
                "title": reader.metadata.title if reader.metadata else None,
            },
            pages=pages_text,
        )

    def _parse_text(self, path: Path, doc_id: str, file_size: int, file_type: str) -> ParsedDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type=file_type,
            file_size_bytes=file_size,
            content=content,
            metadata={
                "word_count": len(content.split()),
                "lines_count": len(content.splitlines()),
            },
            pages=[content],
        )

    def _parse_csv(self, path: Path, doc_id: str, file_size: int) -> ParsedDocument:
        df = pd.read_csv(path)
        return self._df_to_parsed_doc(df, path.name, doc_id, file_size)

    def _df_to_parsed_doc(self, df: pd.DataFrame, filename: str, doc_id: str, file_size: int) -> ParsedDocument:
        # Convert tabular data into descriptive contextual paragraphs and markdown table
        try:
            markdown_table = df.to_markdown(index=False)
        except Exception:
            # Fallback simple markdown table generator
            headers = "| " + " | ".join(df.columns) + " |"
            sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = ["| " + " | ".join(str(val) for val in row) + " |" for _, row in df.iterrows()]
            markdown_table = "\n".join([headers, sep] + rows)

        row_descriptions = []
        for idx, row in df.iterrows():
            row_items = [f"{col}: {val}" for col, val in row.items()]
            row_descriptions.append(f"Row {idx + 1} -> " + ", ".join(row_items))

        combined_content = (
            f"Table Overview: {filename} containing {len(df)} records across columns: {', '.join(df.columns)}.\n\n"
            f"Markdown Representation:\n{markdown_table}\n\n"
            f"Structured Row Records:\n" + "\n".join(row_descriptions)
        )

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=".csv",
            file_size_bytes=file_size,
            content=combined_content,
            metadata={
                "rows_count": len(df),
                "columns": list(df.columns),
                "word_count": len(combined_content.split()),
            },
            pages=[combined_content],
        )

    def _parse_json(self, path: Path, doc_id: str, file_size: int) -> ParsedDocument:
        content_raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(content_raw)
        formatted = json.dumps(data, indent=2)
        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type=".json",
            file_size_bytes=file_size,
            content=formatted,
            metadata={"keys": list(data.keys()) if isinstance(data, dict) else len(data)},
            pages=[formatted],
        )
