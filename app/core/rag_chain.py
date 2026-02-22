import os
import re
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from app.core.document_loader import Document

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an Enterprise RAG Knowledge Assistant, an elite AI system designed for enterprise-grade Q&A.

CRITICAL INSTRUCTIONS:
1. Answer the user's question STRICTLY using only the provided context snippets below.
2. If the context does not contain enough information to answer, state clearly: "I cannot find sufficient evidence in the indexed enterprise documents to answer this query." Do NOT synthesize external assumptions or fabricate facts.
3. Cite your sources clearly using standard markdown format like [Document Name, Page/Section X].
4. Maintain a professional, executive tone.

=== RETRIEVED CONTEXT SNIPPETS ===
{context_text}
==================================

User Query: {query}

Professional Enterprise Answer:"""

class RAGPipelineChain:
    """Enterprise RAG Chain managing prompt engineering, LLM inference, citations, and hallucination guardrails."""

    def __init__(self, vector_store, reranker):
        self.vector_store = vector_store
        self.reranker = reranker

    def run_query(
        self,
        query: str,
        top_k_retrieval: int = 5,
        enable_reranking: bool = True,
        top_k_reranked: int = 3,
        llm_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Retrieval
        initial_retrieved = self.vector_store.similarity_search(query, top_k=top_k_retrieval)

        # Step 2: Context Re-Ranking
        if enable_reranking and initial_retrieved:
            rerank_output = self.reranker.rerank(query, initial_retrieved, top_n=top_k_reranked)
            final_context_items = rerank_output["reranked_results"]
            precision_improvement = rerank_output["precision_improvement_pct"]
        else:
            final_context_items = initial_retrieved[:top_k_reranked]
            precision_improvement = 0.0

        # Step 3: Context Assembly & Citation Metadata
        context_blocks = []
        citations = []
        for idx, (doc, score) in enumerate(final_context_items):
            source = doc.metadata.get("source", "Enterprise Document")
            page_info = f"Page {doc.metadata.get('page')}" if "page" in doc.metadata else f"Chunk {doc.metadata.get('chunk_index', idx+1)}"
            context_blocks.append(f"[{idx+1}] Source: {source} ({page_info})\nRelevance Score: {score:.3f}\nContent:\n{doc.content}")
            citations.append({
                "citation_id": idx + 1,
                "source": source,
                "location": page_info,
                "score": round(score, 4),
                "snippet": doc.content[:250] + "..." if len(doc.content) > 250 else doc.content
            })

        full_context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant enterprise context retrieved."

        # Step 4: Prompt Construction & LLM Generation
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context_text=full_context_str, query=query)
        
        answer, provider_used = self._generate_answer(prompt, final_context_items, query, llm_provider)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": answer,
            "latency_ms": latency_ms,
            "llm_provider": provider_used,
            "vector_backend": self.vector_store.backend_type.upper(),
            "precision_improvement_pct": precision_improvement,
            "citations": citations,
            "retrieved_context_count": len(final_context_items)
        }

    def _generate_answer(
        self,
        prompt: str,
        context_items: List[Tuple[Document, float]],
        query: str,
        override_provider: Optional[str]
    ) -> Tuple[str, str]:
        provider = (override_provider or os.getenv("LLM_PROVIDER", "local_huggingface")).lower()

        # 1. Gemini API if configured
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if (provider == "gemini" or not provider) and gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel(os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash"))
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip(), "Google Gemini API"
            except Exception as e:
                logger.warning(f"Gemini API error ({e}). Falling back to local generation engine.")

        # 2. OpenAI API if configured
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if provider == "openai" and openai_key:
            try:
                import requests
                headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=10)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip(), "OpenAI API"
            except Exception as e:
                logger.warning(f"OpenAI API error ({e}). Falling back to local generation engine.")

        # 3. Grounded Local Generation Engine (Direct Q&A synthesis)
        return self._local_grounded_synthesizer(query, context_items), "Enterprise Grounded Q&A Synthesizer"

    def _local_grounded_synthesizer(self, query: str, context_items: List[Tuple[Document, float]]) -> str:
        if not context_items:
            return "I cannot find sufficient evidence in the indexed enterprise documents to answer this query."

        top_doc, top_score = context_items[0]
        source = top_doc.metadata.get("source", "Enterprise Document")
        location = f"Page {top_doc.metadata.get('page')}" if "page" in top_doc.metadata else "Section 1"

        query_lower = query.lower()
        full_text = "\n".join([doc.content for doc, _ in context_items])

        # Category 1: Technical Skills / Resume Skills
        if any(term in query_lower for term in ["skill", "skills", "technical", "expertise", "technology", "technologies", "languages", "tools"]):
            skills_extracted = self._extract_skills_from_text(full_text)
            if skills_extracted:
                formatted_skills = "\n".join([f"• {s}" for s in skills_extracted])
                return (
                    f"**Direct Answer:**\n"
                    f"Based on `{source}`, here are the technical skills and tools identified:\n\n"
                    f"{formatted_skills}\n\n"
                    f"--- \n"
                    f"📌 **Source:** `{source}` ({location}) | **Confidence:** `{top_score * 100:.1f}%`"
                )

        # Category 2: Data Encryption & Security Policy
        if any(term in query_lower for term in ["encrypt", "encryption", "security", "cipher", "tls", "aes"]):
            sec_lines = [l.strip() for l in full_text.split("\n") if any(k in l.lower() for k in ["encrypt", "aes", "tls", "key", "security", "cipher", "kms"])]
            if sec_lines:
                formatted_sec = "\n".join([f"• {l}" for l in sec_lines[:6]])
                return (
                    f"**Direct Answer:**\n"
                    f"Here are the enterprise data encryption and cybersecurity standards specified in `{source}`:\n\n"
                    f"{formatted_sec}\n\n"
                    f"--- \n"
                    f"📌 **Source:** `{source}` ({location}) | **Confidence:** `{top_score * 100:.1f}%`"
                )

        # Category 3: Database SLAs, RTO, RPO, Cloud Specs
        if any(term in query_lower for term in ["rto", "rpo", "sla", "redundancy", "failover", "database", "redis", "region"]):
            sla_lines = [l.strip() for l in full_text.split("\n") if any(k in l.lower() for k in ["rpo", "rto", "sla", "failover", "redundancy", "replica", "utilization", "region"])]
            if sla_lines:
                formatted_sla = "\n".join([f"• {l}" for l in sla_lines[:6]])
                return (
                    f"**Direct Answer:**\n"
                    f"Here are the architecture metrics and SLAs specified in `{source}`:\n\n"
                    f"{formatted_sla}\n\n"
                    f"--- \n"
                    f"📌 **Source:** `{source}` ({location}) | **Confidence:** `{top_score * 100:.1f}%`"
                )

        # Category 4: General Question Answering (Sentence Extraction)
        stop_words = {"what", "are", "is", "the", "and", "for", "with", "tell", "about", "how", "does", "where"}
        keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2 and w.lower() not in stop_words]

        sentences = re.split(r'(?<=[.!?])\s+|\n+', full_text)
        relevant_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 10 and any(kw in s_clean.lower() for kw in keywords):
                if s_clean not in relevant_sentences:
                    relevant_sentences.append(s_clean)

        if relevant_sentences:
            formatted_ans = "\n\n".join([f"• {s}" for s in relevant_sentences[:5]])
            return (
                f"**Direct Answer:**\n\n"
                f"{formatted_ans}\n\n"
                f"--- \n"
                f"📌 **Source Document:** `{source}` ({location}) | **Confidence Score:** `{top_score * 100:.1f}%`"
            )

        return (
            f"**Direct Answer:**\n\n"
            f"{top_doc.content[:600]}\n\n"
            f"--- \n"
            f"📌 **Source:** `{source}` ({location}) | **Confidence:** `{top_score * 100:.1f}%`"
        )

    def _extract_skills_from_text(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        known_techs = [
            "Python", "SQL", "Java", "C++", "R", "Machine Learning", "Deep Learning",
            "Artificial Intelligence", "Data Science", "PyTorch", "TensorFlow", "Scikit-Learn",
            "Pandas", "NumPy", "FAISS", "ChromaDB", "Hugging Face", "Transformers",
            "LangChain", "FastAPI", "Streamlit", "Docker", "Git", "Linux", "AWS", "GCP", "Kubernetes"
        ]
        
        found_known = []
        text_lower = text.lower()
        for tech in known_techs:
            if tech.lower() in text_lower:
                found_known.append(tech)

        skills = []
        in_skills_section = False
        for line in lines:
            if any(h in line.lower() for h in ["skill", "expertise", "competencies", "technologies", "languages", "proficiencies"]):
                skills.append(f"**{line}**")
                in_skills_section = True
                continue
            if in_skills_section:
                if len(line) < 150 and not line.startswith("http"):
                    skills.append(line)
                if len(skills) > 6:
                    break

        if found_known and len(skills) < 3:
            skills.append(f"**Extracted Technologies & Tools:** {', '.join(found_known)}")

        return skills if skills else [f"**Extracted Technologies & Tools:** {', '.join(found_known)}"] if found_known else lines[:5]
