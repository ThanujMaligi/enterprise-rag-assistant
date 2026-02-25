import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.api_routes import router as api_router

app = FastAPI(
    title=os.getenv("APP_NAME", "Enterprise RAG Knowledge Assistant"),
    description="Production-grade RAG System with FastAPI, LangChain, Transformers, FAISS, ChromaDB, and Context Re-Ranking.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
def root():
    return {
        "service": "Enterprise RAG Knowledge Assistant API",
        "documentation": "/docs",
        "health_check": "/api/v1/health"
    }

if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
