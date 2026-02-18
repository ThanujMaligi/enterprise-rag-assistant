import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.core.document_loader import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Unified Vector Store Manager providing FAISS and ChromaDB backends with live dynamic switching."""

    def __init__(self, embedding_engine, default_backend: str = "faiss", persist_dir: str = "./data"):
        self.embedding_engine = embedding_engine
        self.backend_type = default_backend.lower()
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.documents: List[Document] = []
        self.faiss_index = None
        self.chroma_collection = None

        self._init_backend()

    def _init_backend(self):
        if self.backend_type == "chromadb":
            self._init_chroma()
        else:
            self._init_faiss()

    def set_backend(self, backend_type: str):
        if backend_type.lower() not in ["faiss", "chromadb"]:
            raise ValueError(f"Unsupported vector store backend: {backend_type}")
        self.backend_type = backend_type.lower()
        self._init_backend()
        logger.info(f"Switched vector store backend to: {self.backend_type}")

    def _init_faiss(self):
        try:
            import faiss
            dim = self.embedding_engine.dimension
            self.faiss_index = faiss.IndexFlatL2(dim)
            logger.info(f"Initialized FAISS IndexFlatL2 (dim={dim})")
        except Exception as e:
            logger.warning(f"FAISS init issue ({e}). Using in-memory cosine fallback index.")
            self.faiss_index = None

    def _init_chroma(self):
        try:
            import chromadb
            chroma_dir = os.path.join(self.persist_dir, "chroma_db")
            client = chromadb.PersistentClient(path=chroma_dir)
            self.chroma_collection = client.get_or_create_collection(
                name="enterprise_rag_knowledge",
                metadata={"description": "Enterprise Document Vectors"}
            )
            logger.info(f"Initialized ChromaDB persistent collection at {chroma_dir}")
        except Exception as e:
            logger.warning(f"ChromaDB init issue ({e}). Utilizing fallback vector store engine.")
            self._init_faiss()

    def add_documents(self, docs: List[Document]) -> int:
        if not docs:
            return 0

        texts = [doc.content for doc in docs]
        embeddings = self.embedding_engine.embed_documents(texts)

        start_idx = len(self.documents)
        self.documents.extend(docs)

        if self.backend_type == "faiss" and self.faiss_index is not None:
            self.faiss_index.add(embeddings)
        elif self.backend_type == "chromadb" and self.chroma_collection is not None:
            ids = [doc.doc_id for doc in docs]
            metadatas = [{k: str(v) for k, v in doc.metadata.items()} for doc in docs]
            self.chroma_collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )

        return len(docs)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []

        query_vec = self.embedding_engine.embed_query(query)

        if self.backend_type == "chromadb" and self.chroma_collection is not None and self.chroma_collection.count() > 0:
            results = self.chroma_collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=min(top_k, self.chroma_collection.count())
            )
            retrieved = []
            if results and 'documents' in results and results['documents']:
                retrieved_texts = results['documents'][0]
                distances = results['distances'][0] if 'distances' in results else [0.5]*len(retrieved_texts)
                for text, dist in zip(retrieved_texts, distances):
                    # Match document object
                    matched_doc = next((d for d in self.documents if d.content == text), Document(content=text))
                    similarity_score = max(0.0, float(1.0 / (1.0 + dist)))
                    retrieved.append((matched_doc, similarity_score))
            return retrieved

        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            query_matrix = np.array([query_vec], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_matrix, min(top_k, self.faiss_index.ntotal))
            retrieved = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    score = float(1.0 / (1.0 + dist))
                    retrieved.append((self.documents[idx], score))
            return retrieved

        # In-memory cosine similarity fallback
        doc_vectors = self.embedding_engine.embed_documents([d.content for d in self.documents])
        scores = np.dot(doc_vectors, query_vec) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vec) + 1e-8)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]

    def clear(self):
        self.documents = []
        self._init_backend()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_backend": self.backend_type.upper(),
            "total_documents": len(self.documents),
            "faiss_index_count": self.faiss_index.ntotal if self.faiss_index is not None else 0,
            "chroma_count": self.chroma_collection.count() if self.chroma_collection is not None else 0,
            "embedding_dimension": self.embedding_engine.dimension
        }
