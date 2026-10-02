import re
from dataclasses import dataclass, field
from typing import List, Dict, Any
from rag_assistant.ingestion.parser import ParsedDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """
    Intelligent Recursive Character Chunker with sliding overlap
    and semantic delimiter preservation.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, separators: List[str] | None = None):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Standard rough heuristic: ~1 token per 4 characters or word count * 1.3
        words = len(text.split())
        return max(1, int(words * 1.25))

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        splits = self._recursive_split(text, self.separators)
        chunks = self._merge_splits(splits)
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_pieces: List[str] = []
        if not separators:
            # Base case: split by character if no separators left
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        for split in splits:
            if not split:
                continue
            if len(split) <= self.chunk_size:
                final_pieces.append(split)
            else:
                sub_splits = self._recursive_split(split, remaining_separators)
                final_pieces.extend(sub_splits)

        return final_pieces

    def _merge_splits(self, splits: List[str]) -> List[str]:
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece)
            if current_length + piece_len + 1 > self.chunk_size and current_chunk:
                merged = " ".join(current_chunk).strip()
                if merged:
                    chunks.append(merged)

                # Keep overlap from the end of current chunk
                overlap_pieces: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_pieces.insert(0, item)
                        overlap_len += len(item) + 1
                    else:
                        break

                current_chunk = overlap_pieces
                current_length = sum(len(p) for p in current_chunk) + len(current_chunk)

            current_chunk.append(piece)
            current_length += piece_len + 1

        if current_chunk:
            final_text = " ".join(current_chunk).strip()
            if final_text:
                chunks.append(final_text)

        return chunks

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        raw_chunks = self.split_text(doc.content)
        chunk_objects: List[DocumentChunk] = []

        for idx, text in enumerate(raw_chunks):
            chunk_id = f"{doc.doc_id}_c{idx:03d}"
            tokens = self.estimate_tokens(text)
            chunk_metadata = {
                **doc.metadata,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "chunk_size_chars": len(text),
                "total_chunks_in_doc": len(raw_chunks),
            }

            chunk_objects.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    chunk_index=idx,
                    content=text,
                    token_count=tokens,
                    metadata=chunk_metadata,
                )
            )

        return chunk_objects
