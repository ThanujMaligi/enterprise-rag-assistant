import os
import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Enterprise Embedding Engine using SentenceTransformers with robust vector fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.dimension = 384
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = self.model.get_embedding_dimension()
            else:
                self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Embedding dim: {self.dimension}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ('{self.model_name}'): {e}. Utilizing deterministic semantic hash embeddings.")
            self.model = None

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.model:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return embeddings.astype(np.float32)
            except Exception as e:
                logger.error(f"Error during model encoding: {e}. Falling back to deterministic embeddings.")

        return self._generate_fallback_embeddings(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Deterministic TF-IDF/semantic feature hash fallback embedding generator."""
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for idx, word in enumerate(words):
                h = sum(ord(c) * (i + 1) for i, c in enumerate(word))
                dim_idx = h % self.dimension
                vec[dim_idx] += 1.0 / (1.0 + np.log1p(idx))
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)
