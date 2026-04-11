"""
=============================================================
LEGAL QA SYSTEM — STEP 5 & 6: Retrieval + Strict Answering
=============================================================
MUHIM QOIDALAR:
- Faqat topilgan hujjatlar asosida javob beradi
- Tashqi bilimdan FOYDALANMAYDI
- Javob topilmasa aniq aytadi
- Har bir javob uchun manba ko'rsatadi
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import requests

from .document_loader import DocumentChunk
from .vector_store import LegalVectorStore


# -------------------------------------------------------
# Answer Data Models
# -------------------------------------------------------

class ConfidenceLevel(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NOT_FOUND = "Not Found"


@dataclass
class SourceReference:
    """Manba manzili"""
    doc_name: str
    page_number: int
    section_hint: str
    chunk_index: int
    relevance_score: float
    text_excerpt: str  # Relevant matnning qismi


@dataclass
class LegalAnswer:
    """Huquqiy savol javobi"""
    question: str
    answer: str
    confidence: ConfidenceLevel
    sources: List[SourceReference]
    retrieved_chunks: List[DocumentChunk]
    is_found: bool
    disclaimer: str = (
        "⚠️  DISCLAIMER: This system provides information based ONLY on the "
        "provided legal documents and is NOT legal advice. Always consult a "
        "qualified legal professional for actual legal matters."
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON formatga aylantirish"""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence.value,
            "is_found": self.is_found,
            "sources": [
                {
                    "document": s.doc_name,
                    "page": s.page_number,
                    "section": s.section_hint or "N/A",
                    "relevance_score": round(s.relevance_score, 4),
                    "excerpt": s.text_excerpt,
                }
                for s in self.sources
            ],
            "disclaimer": self.disclaimer,
        }
    
    def format_display(self) -> str:
        """Ekranda chiroyli ko'rsatish uchun formatlash"""
        lines = []
        
        # Sarlavha
        lines.append("=" * 65)
        lines.append("📋  LEGAL QA SYSTEM — JAVOB")
        lines.append("=" * 65)
        
        # Savol
        lines.append(f"\n❓ SAVOL:\n   {self.question}")
        
        # Javob
        lines.append("\n" + "-" * 65)
        
        if self.is_found:
            confidence_emoji = {
                ConfidenceLevel.HIGH: "🟢",
                ConfidenceLevel.MEDIUM: "🟡",
                ConfidenceLevel.LOW: "🔴",
            }.get(self.confidence, "⚪")
            
            lines.append(f"\n💬 JAVOB [{confidence_emoji} {self.confidence.value} ishonch]:\n")
            lines.append(f"   {self.answer}")
            
            # Manbalar
            lines.append("\n" + "-" * 65)
            lines.append(f"\n📚 MANBALAR ({len(self.sources)} ta):\n")
            
            for i, src in enumerate(self.sources, 1):
                lines.append(f"   [{i}] 📄 {src.doc_name}")
                lines.append(f"       📍 Sahifa: {src.page_number}")
                if src.section_hint:
                    lines.append(f"       📑 Bo'lim: {src.section_hint}")
                lines.append(f"       🔍 Relevantlik: {src.relevance_score:.2%}")
                lines.append(f"       💬 Parcha: \"{src.text_excerpt}...\"")
                lines.append("")
        else:
            lines.append("\n❌ JAVOB TOPILMADI\n")
            lines.append(f"   {self.answer}")
        
        # Disclaimer
        lines.append("-" * 65)
        lines.append(f"\n{self.disclaimer}")
        lines.append("=" * 65)
        
        return "\n".join(lines)


# -------------------------------------------------------
# Query Expansion (Multi-Query)
# -------------------------------------------------------

def expand_query(question: str) -> List[str]:
    """
    Savolni bir nechta versiyaga aylantiradi (yaxshiroq qidirish uchun).
    
    Bu oddiy heuristic usul. OpenAI/Claude API mavjud bo'lsa, 
    LLM yordamida ham kengaytirish mumkin.
    
    Args:
        question: Original savol
    
    Returns:
        Kengaytirilgan savollar ro'yxati
    """
    queries = [question]
    
    # "What is" → "definition of" varianti
    if re.match(r'(?i)^(what is|what are)', question):
        alt = re.sub(r'(?i)^(what is|what are)\s+', 'definition of ', question)
        queries.append(alt)
    
    # "How to" → "procedure for" varianti
    if re.match(r'(?i)^how to', question):
        alt = re.sub(r'(?i)^how to\s+', 'procedure for ', question)
        queries.append(alt)
    
    # Savol belgisini olib tashlangan versiya
    plain = re.sub(r'[?!]', '', question).strip()
    if plain != question:
        queries.append(plain)
    
    # Key wordlarni ajratib olish
    # Stop words ni olib tashlash
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what',
                  'how', 'when', 'where', 'who', 'which', 'can', 'does',
                  'do', 'in', 'of', 'to', 'for', 'and', 'or', 'not'}
    
    words = question.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    if len(keywords) >= 2:
        queries.append(' '.join(keywords))
    
    return list(dict.fromkeys(queries))  # Takroriy savollarni olib tashlash


# -------------------------------------------------------
# Confidence Scorer
# -------------------------------------------------------

def calculate_confidence(
    top_scores: List[float],
    answer_text: str,
) -> ConfidenceLevel:
    """
    Javob ishonchlilik darajasini hisoblaydi.
    
    Args:
        top_scores: FAISS similarity score'lari
        answer_text: Berilgan javob matni
    
    Returns:
        ConfidenceLevel enum qiymati
    """
    if not top_scores:
        return ConfidenceLevel.NOT_FOUND
    
    # "Not found" javobini aniqlash
    not_found_phrases = [
        "not found", "not mentioned", "not available", 
        "topilmadi", "yo'q", "mavjud emas"
    ]
    
    if any(phrase in answer_text.lower() for phrase in not_found_phrases):
        return ConfidenceLevel.NOT_FOUND
    
    avg_score = sum(top_scores) / len(top_scores)
    max_score = max(top_scores)
    
    if max_score >= 0.75 and avg_score >= 0.65:
        return ConfidenceLevel.HIGH
    elif max_score >= 0.55 or avg_score >= 0.50:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# -------------------------------------------------------
# Main Answering Engine
# -------------------------------------------------------

class LegalAnsweringEngine:
    """
    Qattiq (strict) huquqiy savol-javob tizimi.
    
    KAFOLAT:
    - Hujjatlar tashqarisidan HECH QACHON javob bermaydi
    - Har bir javob uchun manba ko'rsatadi
    - Kam ishonchli bo'lsa xabar beradi
    """
    
    def __init__(
        self,
        vector_store: LegalVectorStore,
        anthropic_api_key: Optional[str] = None,
        top_k: int = 5,
        use_multi_query: bool = True,
        min_similarity_threshold: float = 0.35,
    ):
        """
        Args:
            vector_store: Tayyor vector store
            anthropic_api_key: Claude API key (agar mavjud bo'lsa)
            top_k: Qidiruvda qaytariladigan chunk'lar soni
            use_multi_query: Ko'p query versiyalarini ishlatish
            min_similarity_threshold: Minimal similarity darajasi
        """
        self.vector_store = vector_store
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.top_k = top_k
        self.use_multi_query = use_multi_query
        self.min_threshold = min_similarity_threshold
    
    def _retrieve_chunks(
        self,
        question: str,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Savolga mos chunk'larni topadi.
        
        Args:
            question: Foydalanuvchi savoli
        
        Returns:
            (chunk, score) juftliklari ro'yxati
        """
        if self.use_multi_query:
            # Bir nechta query versiyalari bilan qidirish
            queries = expand_query(question)
            results = self.vector_store.multi_query_search(
                queries,
                top_k=self.top_k,
            )
        else:
            results = self.vector_store.search(question, top_k=self.top_k)
        
        # Minimal threshold'dan past natijalarni olib tashlash
        filtered = [
            (chunk, score) for chunk, score in results
            if score >= self.min_threshold
        ]
        
        return filtered
    
    def _build_context(self, chunks_with_scores: List[Tuple[DocumentChunk, float]]) -> str:
        """
        Chunk'lardan LLM uchun kontekst matni yasaydi.
        
        Args:
            chunks_with_scores: (chunk, score) juftliklari
        
        Returns:
            Formatlangan kontekst
        """
        context_parts = []
        
        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            section_info = f" | {chunk.section_hint}" if chunk.section_hint else ""
            
            context_parts.append(
                f"[SOURCE {i}]\n"
                f"Document: {chunk.doc_name}\n"
                f"Page: {chunk.page_number}{section_info}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Content:\n{chunk.text}\n"
            )
        
        return "\n" + "—" * 50 + "\n".join(context_parts)
    
    def _build_strict_prompt(self, question: str, context: str) -> str:
        """
        Claude uchun qattiq (strict) prompt yasaydi.
        Bu prompt LLM ni faqat hujjatlar asosida javob berishga majbur qiladi.
        
        Args:
            question: Foydalanuvchi savoli
            context: Qidiruvdan topilgan kontekst
        
        Returns:
            To'liq prompt
        """
        return f"""You are a STRICT legal document analysis system. 

ABSOLUTE RULES (NEVER VIOLATE):
1. Answer ONLY based on the provided source documents below
2. DO NOT use any external knowledge, training data, or general legal knowledge
3. DO NOT guess, infer, or hallucinate information not in the documents
4. If the answer is not explicitly in the documents, respond EXACTLY with: "Answer not found in provided legal documents"
5. Always cite the specific document name and page number for every claim
6. Be precise and quote relevant text when possible

QUESTION:
{question}

PROVIDED LEGAL DOCUMENTS (your ONLY allowed source):
{context}

INSTRUCTIONS FOR YOUR RESPONSE:
1. Search through ALL provided sources carefully
2. If you find relevant information → provide a clear, structured answer with citations
3. If you do NOT find relevant information → respond with "Answer not found in provided legal documents"
4. Format citations as: [Source: Document Name, Page X]
5. Keep your answer factual and precise — no speculation

YOUR ANSWER:"""
    
    def _call_claude_api(
        self,
        prompt: str,
        chunks_with_scores=None,
    ) -> str:
        """Anthropic Claude API ga murojaat qiladi."""
        if not self.api_key:
            return self._heuristic_answer_fallback(chunks_with_scores or [])
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        
        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "system": (
                "You are a strict legal document analyzer. "
                "You ONLY answer based on provided documents. "
                "NEVER use external knowledge. "
                "If answer is not in documents, say exactly: "
                "'Answer not found in provided legal documents'"
            ),
        }
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            data = response.json()
            return data["content"][0]["text"]
        
        except requests.exceptions.Timeout:
            return "API so'rovi vaqt tugadi. Qayta urinib ko'ring."
        except requests.exceptions.RequestException as e:
            return f"API xatosi: {str(e)}"
    
    def _heuristic_answer_fallback(
        self,
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
    ) -> str:
        """
        API key yo'q bo'lganda topilgan chunk'larni to'g'ridan qaytaradi.
        Hujjatdan tashqari hech narsa qo'shilmaydi.
        """
        if not chunks_with_scores:
            return "Answer not found in provided legal documents"

        top = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)[:3]
        parts = ["The following relevant sections were found in the legal documents:\n"]
        for i, (chunk, score) in enumerate(top, 1):
            section = f" — {chunk.section_hint}" if chunk.section_hint else ""
            parts.append(
                f"[{i}] {chunk.doc_name}, Page {chunk.page_number}{section}:\n"
                f"{chunk.text.strip()}\n"
            )
        return "\n".join(parts)

    def answer(self, question: str) -> LegalAnswer:
        """
        Savol berib, faqat hujjatlar asosida javob oladi.
        
        Bu tizimning asosiy funksiyasi.
        
        Args:
            question: Foydalanuvchi savoli
        
        Returns:
            LegalAnswer (javob + manbalar + ishonch darajasi)
        """
        # ── 1. Savolni tekshirish ─────────────────────────────
        if not question or not question.strip():
            return LegalAnswer(
                question=question,
                answer="Bo'sh savol kiritildi.",
                confidence=ConfidenceLevel.NOT_FOUND,
                sources=[],
                retrieved_chunks=[],
                is_found=False,
            )
        
        question = question.strip()
        
        # ── 2. Chunk'larni qidirish ───────────────────────────
        chunks_with_scores = self._retrieve_chunks(question)
        
        if not chunks_with_scores:
            return LegalAnswer(
                question=question,
                answer="Answer not found in provided legal documents. "
                       "No relevant sections were found for your question.",
                confidence=ConfidenceLevel.NOT_FOUND,
                sources=[],
                retrieved_chunks=[],
                is_found=False,
            )
        
        # ── 3. Kontekst yaratish ──────────────────────────────
        context = self._build_context(chunks_with_scores)
        
        # ── 4. LLM javob olish ───────────────────────────────
        prompt = self._build_strict_prompt(question, context)
        raw_answer = self._call_claude_api(prompt, chunks_with_scores)
        
        # ── 5. Javob tahlili ─────────────────────────────────
        not_found_indicators = [
            "not found in provided",
            "not available in",
            "not mentioned in",
            "cannot find",
            "no information",
        ]
        
        is_found = not any(
            indicator in raw_answer.lower()
            for indicator in not_found_indicators
        )
        
        # ── 6. Manbalar ro'yxatini yaratish ───────────────────
        scores = [score for _, score in chunks_with_scores]
        confidence = calculate_confidence(scores, raw_answer)
        
        if not is_found:
            confidence = ConfidenceLevel.NOT_FOUND
        
        sources = [
            SourceReference(
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
                section_hint=chunk.section_hint,
                chunk_index=chunk.chunk_index,
                relevance_score=score,
                text_excerpt=chunk.text[:150].replace("\n", " "),
            )
            for chunk, score in chunks_with_scores
        ]
        
        retrieved_chunks = [chunk for chunk, _ in chunks_with_scores]
        
        return LegalAnswer(
            question=question,
            answer=raw_answer,
            confidence=confidence,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            is_found=is_found,
        )
