import os
import unittest
from app.core.document_loader import EnterpriseDocumentLoader, Document
from app.core.semantic_chunker import SemanticChunker
from app.core.embeddings import EmbeddingEngine
from app.core.vector_store import VectorStoreManager
from app.core.reranker import ContextReranker
from app.core.rag_chain import RAGPipelineChain

class TestEnterpriseRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Enterprise Cloud Architecture Specification v4.2.\n"
            "Active-active deployment across primary us-east-1 and secondary us-west-2 cloud regions.\n"
            "Automated DNS failover under 15 seconds. High availability rate of 99.99%.\n"
            "Data encryption at rest using AES-256 GCM encryption algorithms."
        )
        self.doc = Document(content=self.sample_text, metadata={"source": "test_spec.txt", "page": 1})

    def test_semantic_chunker(self):
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_documents([self.doc])
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].metadata["source"], "test_spec.txt")

    def test_embeddings_and_vector_store_faiss(self):
        embedding_engine = EmbeddingEngine()
        vector_store = VectorStoreManager(embedding_engine=embedding_engine, default_backend="faiss")

        chunker = SemanticChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split_documents([self.doc])

        indexed_count = vector_store.add_documents(chunks)
        self.assertEqual(indexed_count, len(chunks))

        results = vector_store.similarity_search("encryption standards AES-256", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("AES-256", results[0][0].content)

    def test_vector_store_backend_switch(self):
        embedding_engine = EmbeddingEngine()
        vector_store = VectorStoreManager(embedding_engine=embedding_engine, default_backend="faiss")

        # Test switching backend
        vector_store.set_backend("chromadb")
        self.assertEqual(vector_store.backend_type, "chromadb")

        vector_store.set_backend("faiss")
        self.assertEqual(vector_store.backend_type, "faiss")

    def test_reranker_precision_improvement(self):
        reranker = ContextReranker(use_cross_encoder=False)
        items = [
            (Document(content="General server setup instructions and cloud guidelines."), 0.45),
            (Document(content="Data encryption at rest using AES-256 GCM encryption algorithms."), 0.65)
        ]

        reranked_output = reranker.rerank("What encryption algorithm is used?", items, top_n=2)
        self.assertIn("reranked_results", reranked_output)
        top_doc = reranked_output["reranked_results"][0][0]
        self.assertIn("AES-256", top_doc.content)

    def test_end_to_end_rag_chain(self):
        embedding_engine = EmbeddingEngine()
        vector_store = VectorStoreManager(embedding_engine=embedding_engine, default_backend="faiss")
        chunker = SemanticChunker(chunk_size=150, chunk_overlap=20)
        vector_store.add_documents(chunker.split_documents([self.doc]))

        reranker = ContextReranker(use_cross_encoder=False)
        chain = RAGPipelineChain(vector_store=vector_store, reranker=reranker)

        result = chain.run_query("What encryption algorithm is mandated for data at rest?")
        self.assertIn("answer", result)
        self.assertIn("citations", result)
        self.assertTrue(result["latency_ms"] >= 0)
        self.assertTrue(len(result["citations"]) > 0)

if __name__ == "__main__":
    unittest.main()
