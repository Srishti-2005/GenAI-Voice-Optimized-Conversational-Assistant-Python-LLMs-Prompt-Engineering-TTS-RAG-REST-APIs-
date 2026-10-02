from typing import List, Dict, Any


class RAGPromptTemplate:
    """Production prompt engineering template for grounded document Q&A."""

    DEFAULT_SYSTEM_PROMPT = """You are an intelligent, precise Enterprise Document AI Assistant.
Your mission is to answer user inquiries strictly and accurately using the provided enterprise context snippets.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer strictly using facts found directly in the retrieved context. Do NOT fabricate or assume information outside the provided text.
2. Citations: After each substantive claim or fact, cite the relevant source using format: [Source: <filename> | Chunk: <chunk_id>].
3. Transparency: If the context snippets do not contain sufficient evidence to answer the question, clearly state:
   "Based on the provided enterprise documents, there is insufficient information to answer this query."
4. Structure: Structure your answer cleanly with an executive summary, clear bullet points, and citation tags.
"""

    @classmethod
    def format_context(cls, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant documents found in repository."

        context_blocks = []
        for idx, chunk in enumerate(chunks):
            filename = chunk.get("filename", "unknown")
            chunk_id = chunk.get("chunk_id", f"c_{idx}")
            content = chunk.get("content", "").strip()
            score = chunk.get("similarity_score") or chunk.get("final_score") or 0.0

            block = (
                f"--- CONTEXT SNIPPET #{idx + 1} ---\n"
                f"Document: {filename} (ID: {chunk_id}, Relevance: {score:.3f})\n"
                f"Content:\n{content}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    @classmethod
    def build_user_prompt(cls, query: str, chunks: List[Dict[str, Any]]) -> str:
        context_str = cls.format_context(chunks)
        return (
            f"ENTERPRISE CONTEXT:\n{context_str}\n\n"
            f"USER QUERY: {query}\n\n"
            f"Please provide an accurate, well-cited response following the guidelines above:"
        )
