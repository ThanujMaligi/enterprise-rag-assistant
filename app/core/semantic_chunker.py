import re
from typing import List, Dict, Any
from app.core.document_loader import Document

class SemanticChunker:
    """Semantic Chunker splitting enterprise text into meaningful vector-indexable segments."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, strategy: str = "recursive"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for idx, chunk_text in enumerate(chunks):
                meta = doc.metadata.copy()
                meta.update({
                    "chunk_id": f"{doc.doc_id}_chunk_{idx}",
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk_text)
                })
                chunked_docs.append(Document(
                    content=chunk_text,
                    metadata=meta,
                    doc_id=f"{doc.doc_id}_{idx}"
                ))
        return chunked_docs

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        if self.strategy == "semantic_paragraphs":
            return self._split_by_paragraphs(text)
        else:
            return self._recursive_split(text)

    def _recursive_split(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        return self._split_text_with_separators(text, separators, self.chunk_size, self.chunk_overlap)

    def _split_text_with_separators(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        separator = separators[-1]
        for s in separators:
            if s in text:
                separator = s
                break

        if separator == "":
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - chunk_overlap
            return chunks

        splits = text.split(separator)
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split) + len(separator)
            if current_length + split_len > chunk_size and current_chunk:
                merged = separator.join(current_chunk).strip()
                if merged:
                    chunks.append(merged)
                # handle overlap
                overlap_size = 0
                new_chunk = []
                for prev in reversed(current_chunk):
                    if overlap_size + len(prev) <= chunk_overlap:
                        new_chunk.insert(0, prev)
                        overlap_size += len(prev)
                    else:
                        break
                current_chunk = new_chunk
                current_length = sum(len(p) for p in current_chunk) + (len(current_chunk) * len(separator))

            current_chunk.append(split)
            current_length += split_len

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks

    def _split_by_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = ""

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            if len(current) + len(p_clean) + 2 <= self.chunk_size:
                current = f"{current}\n\n{p_clean}" if current else p_clean
            else:
                if current:
                    chunks.append(current)
                current = p_clean

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:self.chunk_size]]
