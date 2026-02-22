import re
import numpy as np
from typing import List, Tuple, Dict, Any
from app.core.document_loader import Document

class ContextReranker:
    """Context Re-ranking Engine combining lexical overlap, semantic alignment, and cross-attention scoring."""

    def __init__(self, use_cross_encoder: bool = True):
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model = None
        self._init_cross_encoder()

    def _init_cross_encoder(self):
        if self.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            except Exception:
                self.cross_encoder_model = None

    def rerank(
        self,
        query: str,
        retrieved_items: List[Tuple[Document, float]],
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Re-ranks vector retrieved items and returns re-ordered candidates along with
        dynamically computed precision telemetry.
        """
        if not retrieved_items:
            return {
                "reranked_results": [],
                "precision_improvement_pct": 0.0,
                "initial_avg_score": 0.0,
                "reranked_avg_score": 0.0
            }

        docs = [item[0] for item in retrieved_items]
        initial_scores = [item[1] for item in retrieved_items]

        if self.cross_encoder_model is not None:
            pairs = [[query, doc.content] for doc in docs]
            try:
                raw_scores = self.cross_encoder_model.predict(pairs)
                # Normalize scores with sigmoid
                reranked_scores = [float(1.0 / (1.0 + np.exp(-s))) for s in raw_scores]
            except Exception:
                reranked_scores = self._composite_rerank_scores(query, retrieved_items)
        else:
            reranked_scores = self._composite_rerank_scores(query, retrieved_items)

        # Pair documents with new reranked scores
        combined = list(zip(docs, reranked_scores))
        # Sort by reranked score descending
        combined.sort(key=lambda x: x[1], reverse=True)
        top_reranked = combined[:top_n]

        # Calculate dynamic precision improvement
        initial_avg = float(np.mean(initial_scores)) if initial_scores else 0.0
        reranked_avg = float(np.mean([s for _, s in top_reranked])) if top_reranked else 0.0
        
        improvement_pct = 0.0
        if initial_avg > 0:
            improvement_pct = max(0.0, ((reranked_avg - initial_avg) / initial_avg) * 100.0)

        return {
            "reranked_results": top_reranked,
            "precision_improvement_pct": round(improvement_pct, 1),
            "initial_avg_score": round(initial_avg, 4),
            "reranked_avg_score": round(reranked_avg, 4)
        }

    def _composite_rerank_scores(self, query: str, items: List[Tuple[Document, float]]) -> List[float]:
        """Fallback hybrid ranker combining dense vector similarity with lexical BM25-style keyword matching."""
        query_words = set(re.findall(r'\w+', query.lower()))
        scores = []

        for doc, vector_score in items:
            doc_words = re.findall(r'\w+', doc.content.lower())
            if not doc_words:
                scores.append(vector_score)
                continue

            matches = sum(1 for w in query_words if w in doc_words)
            lexical_score = matches / max(1, len(query_words))
            
            # Weighted hybrid blend
            hybrid_score = (0.55 * vector_score) + (0.45 * lexical_score)
            scores.append(hybrid_score)

        return scores
