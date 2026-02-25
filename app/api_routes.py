import os
import shutil
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from app.core.document_loader import EnterpriseDocumentLoader, Document
from app.core.semantic_chunker import SemanticChunker
from app.core.embeddings import EmbeddingEngine
from app.core.vector_store import VectorStoreManager
from app.core.reranker import ContextReranker
from app.core.rag_chain import RAGPipelineChain

router = APIRouter(prefix="/api/v1", tags=["Enterprise RAG Endpoints"])

# Global Service Instances
embedding_engine = EmbeddingEngine()
vector_store = VectorStoreManager(embedding_engine=embedding_engine)
reranker = ContextReranker()
rag_chain = RAGPipelineChain(vector_store=vector_store, reranker=reranker)

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the encryption standards specified in our security policy?")
    top_k_retrieval: int = Field(5, ge=1, le=20)
    top_k_reranked: int = Field(3, ge=1, le=10)
    enable_reranking: bool = Field(True)
    vector_backend: Optional[str] = Field("faiss", description="Choose 'faiss' or 'chromadb'")
    llm_provider: Optional[str] = Field(None, description="Optional override: 'gemini', 'openai', or 'local_huggingface'")

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "system": "Enterprise RAG Knowledge Assistant",
        "version": "2.0.0",
        "vector_store_stats": vector_store.get_stats()
    }

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    vector_backend: str = Form("faiss")
):
    if vector_backend.lower() in ["faiss", "chromadb"]:
        vector_store.set_backend(vector_backend.lower())

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Load document
        loaded_docs = EnterpriseDocumentLoader.load_file(temp_path)
        if not loaded_docs:
            raise HTTPException(status_code=400, detail="Failed to parse document or document is empty.")

        # 2. Semantic Chunking
        chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_docs = chunker.split_documents(loaded_docs)

        # 3. Vector Indexing
        indexed_count = vector_store.add_documents(chunked_docs)

        return {
            "message": f"Successfully indexed '{file.filename}'",
            "filename": file.filename,
            "raw_documents_loaded": len(loaded_docs),
            "chunks_created": len(chunked_docs),
            "chunks_indexed": indexed_count,
            "vector_backend": vector_store.backend_type.upper()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@router.post("/query")
def execute_rag_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    if request.vector_backend:
        vector_store.set_backend(request.vector_backend)

    try:
        response = rag_chain.run_query(
            query=request.query,
            top_k_retrieval=request.top_k_retrieval,
            enable_reranking=request.enable_reranking,
            top_k_reranked=request.top_k_reranked,
            llm_provider=request.llm_provider
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")

@router.get("/documents")
def list_documents():
    return {
        "stats": vector_store.get_stats(),
        "document_sources": list(set(d.metadata.get("source", "Unknown") for d in vector_store.documents))
    }

@router.post("/clear")
def clear_vector_store():
    vector_store.clear()
    return {"message": "Vector store and document index cleared successfully."}
