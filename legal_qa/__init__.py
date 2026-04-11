"""
=============================================================
LEGAL QA SYSTEM — STEP 8: Main Automation Pipeline
=============================================================
build_legal_model(folder_path) — barcha qadamlarni avtomatik bajaradi
ask_question(question)          — sodda interfeys

ISHLATISH:
    from legal_qa import build_legal_model
    
    qa = build_legal_model("path/to/legal/documents")
    result = qa.ask_question("What are the contract termination conditions?")
    print(result)
"""

import os
import time
from pathlib import Path
from typing import Optional, List

from .core.document_loader import load_documents_from_folder
from .core.chunker import create_chunks_from_documents, ChunkConfig
from .core.vector_store import LegalVectorStore, EmbeddingEngine
from .core.answering_engine import LegalAnsweringEngine, LegalAnswer


# -------------------------------------------------------
# Legal QA System — Unified Interface
# -------------------------------------------------------

class LegalQASystem:
    """
    Huquqiy savol-javob tizimi — yagona interfeys.
    
    Bu klass barcha komponentlarni birlashtiradi:
    1. Hujjat yuklash
    2. Matn bo'laklash
    3. Vektor ma'lumotlar bazasi
    4. Qattiq javob berish
    
    Misol:
        qa = build_legal_model("./documents")
        result = qa.ask_question("termination conditions?")
        result.format_display()  # Ekranda ko'rsatish
    """
    
    def __init__(
        self,
        vector_store: LegalVectorStore,
        answering_engine: LegalAnsweringEngine,
        document_names: List[str],
        total_chunks: int,
        build_time: float,
    ):
        self.vector_store = vector_store
        self.answering_engine = answering_engine
        self.document_names = document_names
        self.total_chunks = total_chunks
        self.build_time = build_time
    
    def ask_question(self, question: str) -> LegalAnswer:
        """
        Huquqiy savol beradi.
        
        Args:
            question: Sizning savolingiz (ingliz yoki o'zbek tilida)
        
        Returns:
            LegalAnswer: Javob, manbalar va ishonch darajasi
        """
        return self.answering_engine.answer(question)
    
    def ask_and_print(self, question: str) -> LegalAnswer:
        """
        Savol beradi va javobni ekranda ko'rsatadi.
        
        Args:
            question: Savol
        
        Returns:
            LegalAnswer obyekti
        """
        answer = self.ask_question(question)
        print(answer.format_display())
        return answer
    
    def get_info(self) -> str:
        """Tizim haqida ma'lumot"""
        return (
            f"\n{'=' * 55}\n"
            f"⚖️  LEGAL QA SYSTEM — Tayyor!\n"
            f"{'=' * 55}\n"
            f"📂 Hujjatlar soni:  {len(self.document_names)} ta\n"
            f"📄 Hujjat nomlari:\n"
            + "\n".join(f"    • {name}" for name in self.document_names)
            + f"\n✂️  Jami chunk'lar:  {self.total_chunks} ta\n"
            f"⏱️  Qurilish vaqti: {self.build_time:.1f} soniya\n"
            f"{'=' * 55}\n"
        )
    
    def save(self, save_dir: str) -> None:
        """
        Tizimni keyingi sessiyalar uchun saqlaydi.
        
        Args:
            save_dir: Saqlash papkasi
        """
        self.vector_store.save(save_dir)
        print(f"✅ Tizim saqlandi: {save_dir}")
        print("   Keyingi safar load_legal_model() ishlatib yuklang.")
    
    @classmethod
    def load(
        cls,
        save_dir: str,
        anthropic_api_key: Optional[str] = None,
    ) -> "LegalQASystem":
        """
        Saqlangan tizimni yuklaydi (hujjatlarni qayta ishlash shart emas).
        
        Args:
            save_dir: Saqlangan tizim papkasi
            anthropic_api_key: Claude API key
        
        Returns:
            LegalQASystem
        """
        vector_store = LegalVectorStore.load(save_dir)
        
        answering_engine = LegalAnsweringEngine(
            vector_store=vector_store,
            anthropic_api_key=anthropic_api_key,
        )
        
        doc_names = list(set(
            chunk.doc_name for chunk in vector_store.chunks
        ))
        
        return cls(
            vector_store=vector_store,
            answering_engine=answering_engine,
            document_names=doc_names,
            total_chunks=len(vector_store.chunks),
            build_time=0.0,
        )


# -------------------------------------------------------
# Main Build Function
# -------------------------------------------------------

def build_legal_model(
    folder_path: str,
    anthropic_api_key: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    top_k: int = 5,
    embedding_model: str = "local",
    save_dir: Optional[str] = None,
    min_similarity_threshold: float = 0.35,
) -> LegalQASystem:
    """
    ================================================================
    ASOSIY FUNKSIYA: To'liq huquqiy QA tizimini avtomatik quradi.
    ================================================================
    
    Bu funksiya quyidagilarni bajaradi:
    1. Papkadagi barcha PDF/DOCX/TXT fayllarni yuklaydi
    2. Matnlarni aqlli bo'laklarga ajratadi (overlap bilan)
    3. Sentence Transformer yordamida embedding'lar yaratadi
    4. FAISS vektor ma'lumotlar bazasiga saqlaydi
    5. Qattiq javob berish tizimini sozlaydi
    
    Args:
        folder_path: Huquqiy hujjatlar papkasi yo'li
                     (PDF, DOCX, TXT formatlar qo'llab-quvvatlanadi)
        
        anthropic_api_key: Anthropic Claude API key
                           (None bo'lsa heuristic rejimda ishlaydi)
        
        chunk_size: Har bir bo'lak hajmi (belgilar soni)
                    Standart: 800, Tavsiya: 500-1000
        
        chunk_overlap: Qo'shni bo'laklar orasidagi umumiy belgilar
                       Standart: 150, Tavsiya: 100-200
        
        top_k: Har bir savol uchun qidirib topiladigan bo'laklar soni
               Standart: 5, Tavsiya: 3-7
        
        embedding_model: SentenceTransformers model nomi
                         Standart: 'all-MiniLM-L6-v2' (tez va aniq)
                         Yaxshiroq: 'all-mpnet-base-v2' (sekinroq lekin aniqroq)
        
        save_dir: Tizimni saqlash papkasi (None bo'lsa saqlanmaydi)
        
        min_similarity_threshold: Minimal o'xshashlik darajasi
                                   (Bu dan past natijalar o'tkazib yuboriladi)
    
    Returns:
        LegalQASystem: Tayyor, ishlatishga tayyor tizim
    
    Misol:
        # Oddiy ishlatish
        qa = build_legal_model("./contracts")
        answer = qa.ask_question("What is the payment schedule?")
        print(answer.format_display())
        
        # API key bilan (aniqroq javoblar)
        qa = build_legal_model(
            "./contracts",
            anthropic_api_key="your-api-key",
            save_dir="./saved_model"  # Keyingi safar tez yuklash uchun
        )
    
    Raises:
        FileNotFoundError: Papka topilmasa
        ValueError: Papkada hujjat bo'lmasa
    """
    
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("⚖️  LEGAL QA SYSTEM — Qurilmoqda...")
    print("=" * 60)
    
    # ── STEP 1: Hujjatlarni yuklash ───────────────────────────
    print("\n📥 QADAM 1: Hujjatlar yuklanmoqda...")
    documents = load_documents_from_folder(folder_path)
    
    # ── STEP 2: Matnni bo'laklash ─────────────────────────────
    print("\n✂️  QADAM 2: Matn bo'laklarga ajratilmoqda...")
    chunk_config = ChunkConfig(
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    chunks = create_chunks_from_documents(documents, chunk_config)
    
    # ── STEP 3: Embedding engine ──────────────────────────────
    print("\n🧠 QADAM 3: Embedding modeli yuklanmoqda...")
    embedding_engine = EmbeddingEngine(model_name=embedding_model, embedding_dim=128)
    
    # ── STEP 4: Vektor ma'lumotlar bazasi ────────────────────
    print("\n🗄️  QADAM 4: Vektor ma'lumotlar bazasi qurilmoqda...")
    vector_store = LegalVectorStore(embedding_engine)
    vector_store.build(chunks)
    
    # ── STEP 5 & 6: Javob berish tizimi ──────────────────────
    print("\n⚙️  QADAM 5: Javob berish tizimi sozlanmoqda...")
    
    answering_engine = LegalAnsweringEngine(
        vector_store=vector_store,
        anthropic_api_key=anthropic_api_key,
        top_k=top_k,
        use_multi_query=True,
        min_similarity_threshold=min_similarity_threshold,
    )
    
    # ── Saqlash (agar talab qilinsa) ──────────────────────────
    if save_dir:
        print(f"\n💾 Tizim saqlanmoqda: {save_dir}")
        vector_store.save(save_dir)
    
    # ── Tayyor tizim ──────────────────────────────────────────
    build_time = time.time() - start_time
    doc_names = [doc.name for doc in documents]
    
    qa_system = LegalQASystem(
        vector_store=vector_store,
        answering_engine=answering_engine,
        document_names=doc_names,
        total_chunks=len(chunks),
        build_time=build_time,
    )
    
    print(qa_system.get_info())
    
    return qa_system


def load_legal_model(
    save_dir: str,
    anthropic_api_key: Optional[str] = None,
) -> LegalQASystem:
    """
    Oldin saqlangan tizimni tez yuklaydi.
    Hujjatlarni qayta ishlash shart emas!
    
    Args:
        save_dir: build_legal_model() da ko'rsatilgan save_dir
        anthropic_api_key: Claude API key
    
    Returns:
        LegalQASystem: Tayyor tizim
    
    Misol:
        qa = load_legal_model("./saved_model")
        answer = qa.ask_question("What is the penalty clause?")
    """
    return LegalQASystem.load(save_dir, anthropic_api_key)
