"""
=============================================================
LEGAL QA SYSTEM — STEP 3 & 4: Embeddings + Vector Database
=============================================================
Chunk'larni vektorga aylantiradi va FAISS da saqlaydi.

Ikki rejim:
  1. LOCAL (internet shart emas): TF-IDF + SVD — tez, yengil
  2. NEURAL (internet kerak, birinchi marta): SentenceTransformers

Standart: LOCAL rejim (deploy uchun qulay)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .document_loader import DocumentChunk


# -------------------------------------------------------
# Local TF-IDF + SVD Embedding Engine (Internet shart emas)
# -------------------------------------------------------

class EmbeddingEngine:
    """
    Matnlarni vektorlarga aylantiruvchi motor.

    Ikkita strategiya:
      - 'local' : TF-IDF + TruncatedSVD  (internet kerak emas, tez)
      - 'neural': SentenceTransformers   (birinchi marta internet kerak)

    Huquqiy hujjatlar uchun 'local' rejim juda yaxshi ishlaydi
    chunki aniq kalit so'zlar (article, clause, penalty...) muhim.
    """

    def __init__(
        self,
        model_name: str = "local",   # 'local' yoki SentenceTransformer nomi
        embedding_dim: int = 128,    # LOCAL rejim uchun vektor o'lchami
    ):
        self.model_name = model_name
        self._use_local = (model_name == "local")
        self._fitted = False

        # LOCAL rejim komponentlari
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._actual_dim: int = embedding_dim
        self.embedding_dim: int = embedding_dim

        if self._use_local:
            print("\n🤖 Embedding rejimi: LOCAL (TF-IDF + SVD)")
            print("   ✅ Internet kerak emas, to'liq offline ishlaydi")
        else:
            # Neural rejim — lazy import
            print(f"\n🤖 Embedding modeli yuklanmoqda: {model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._neural_model = SentenceTransformer(model_name)
                self.embedding_dim = self._neural_model.get_sentence_embedding_dimension()
                self._actual_dim = self.embedding_dim
                print(f"   ✅ Neural model tayyor! Dim: {self.embedding_dim}")
            except Exception as e:
                print(f"   ⚠️  Neural model yuklanmadi ({e}). LOCAL rejimga o'tildi.")
                self._use_local = True
                self.model_name = "local"

    # ── LOCAL: corpus ustida fit ──────────────────────────
    def fit(self, texts: List[str]) -> None:
        """
        TF-IDF + SVD ni corpus (barcha chunk'lar) ustida o'rgatadi.
        Faqat LOCAL rejimda chaqiriladi.
        """
        if not self._use_local:
            return  # Neural model pre-trained, fit shart emas

        print(f"   📐 TF-IDF o'rgatilmoqda ({len(texts)} ta matn)...")

        self._tfidf = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),       # Unigram + bigram (juft so'zlar)
            sublinear_tf=True,        # log(1+tf) — chastota normalizatsiyasi
            min_df=1,                 # Kamida 1 ta hujjatda bo'lsin
            strip_accents='unicode',
            analyzer='word',
        )

        X_sparse = self._tfidf.fit_transform(texts)
        n_features = X_sparse.shape[1]
        n_samples = X_sparse.shape[0]

        # SVD o'lchami: min(so'raladigan, mavjud-1)
        target_dim = self.embedding_dim
        actual_dim = min(target_dim, n_features - 1, n_samples - 1)
        actual_dim = max(actual_dim, 2)  # Minimal 2

        self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        self._svd.fit(X_sparse)
        self._actual_dim = actual_dim
        self.embedding_dim = actual_dim

        explained = self._svd.explained_variance_ratio_.sum()
        print(f"   ✅ TF-IDF features: {n_features}, SVD dim: {actual_dim}, "
              f"variance explained: {explained:.1%}")
        self._fitted = True

    # ── encode: texts → dense vectors ────────────────────
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Matnlar ro'yxatini vektorlarga aylantiradi."""
        if self._use_local:
            if not self._fitted:
                raise RuntimeError("fit() avval chaqirilmagan!")
            X_sparse = self._tfidf.transform(texts)
            X = self._svd.transform(X_sparse).astype(np.float32)
            return normalize(X, norm='l2')
        else:
            emb = self._neural_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return emb.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Bitta matnni vektorga aylantiradi (query uchun)."""
        return self.encode([text], show_progress=False)

    # ── Saqlash / yuklash ─────────────────────────────────
    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "use_local": self._use_local,
            "fitted": self._fitted,
        }
        with open(path / "engine_state.json", "w") as f:
            json.dump(state, f)
        if self._use_local and self._fitted:
            with open(path / "tfidf.pkl", "wb") as f:
                pickle.dump(self._tfidf, f)
            with open(path / "svd.pkl", "wb") as f:
                pickle.dump(self._svd, f)

    @classmethod
    def load(cls, save_dir: str) -> "EmbeddingEngine":
        path = Path(save_dir)
        with open(path / "engine_state.json") as f:
            state = json.load(f)
        engine = cls(
            model_name=state["model_name"],
            embedding_dim=state["embedding_dim"],
        )
        if state.get("use_local") and state.get("fitted"):
            with open(path / "tfidf.pkl", "rb") as f:
                engine._tfidf = pickle.load(f)
            with open(path / "svd.pkl", "rb") as f:
                engine._svd = pickle.load(f)
            engine._actual_dim = state["embedding_dim"]
            engine._fitted = True
        return engine


# -------------------------------------------------------
# FAISS Vector Store
# -------------------------------------------------------

class LegalVectorStore:
    """
    Huquqiy hujjatlar uchun vektor ma'lumotlar bazasi.
    FAISS (Facebook AI Similarity Search) asosida.
    
    Nima uchun FAISS?
    - Mahalliy ishlaydi (internet kerak emas)
    - Juda tez (millionlab vektorlarda ham)
    - Bepul va open-source
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        """
        Args:
            embedding_engine: Embedding modeli
        """
        self.embedding_engine = embedding_engine
        self.index: Optional[faiss.Index] = None
        self.chunks: List[DocumentChunk] = []
        self.metadata: List[Dict[str, Any]] = []
        self.is_built = False
    
    def build(self, chunks: List[DocumentChunk], batch_size: int = 32) -> None:
        """
        Chunk'lardan vektor indeks quradi.
        
        Args:
            chunks: Hujjat bo'laklari ro'yxati
            batch_size: Embedding batch hajmi
        """
        if not chunks:
            raise ValueError("Chunk'lar ro'yxati bo'sh!")
        
        print(f"\n🔨 Vektor indeks qurilmoqda ({len(chunks)} ta chunk)...")
        
        # Matnlarni ajratib olish
        texts = [chunk.text for chunk in chunks]

        # LOCAL rejim: avval corpus ustida fit qilish kerak
        if self.embedding_engine._use_local:
            self.embedding_engine.fit(texts)

        # Embedding yaratish
        print("   📐 Embedding'lar hisoblanmoqda...")
        embeddings = self.embedding_engine.encode(
            texts,
            batch_size=batch_size,
            show_progress=True,
        )
        
        # FAISS indeks yaratish
        # IndexFlatIP — Inner Product (cosine similarity uchun, normalize qilingan vektorlarda)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        # Vektorlarni qo'shish
        self.index.add(embeddings)
        
        # Chunk'larni va metadata'ni saqlash
        self.chunks = chunks
        self.metadata = self._build_metadata(chunks)
        self.is_built = True
        
        print(f"   ✅ Indeks tayyor! {self.index.ntotal} ta vektor saqlandi.")
    
    def _build_metadata(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Chunk'lardan metadata yaratadi"""
        return [
            {
                "doc_name": chunk.doc_name,
                "doc_path": chunk.doc_path,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "section_hint": chunk.section_hint,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "text_preview": chunk.text[:100] + "...",
            }
            for chunk in chunks
        ]
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Savol bo'yicha eng mos chunk'larni topadi.
        
        Args:
            query: Foydalanuvchi savoli
            top_k: Qaytariladigan natijalar soni
        
        Returns:
            (chunk, similarity_score) tuple'lari ro'yxati
        """
        if not self.is_built:
            raise RuntimeError("Vector store hali qurilmagan! Avval .build() chaqiring.")
        
        # Query vektorini hisoblash
        query_embedding = self.embedding_engine.encode_single(query)
        
        # FAISS qidiruvi
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Haqiqiy indeks
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))
        
        return results
    
    def multi_query_search(
        self,
        queries: List[str],
        top_k: int = 5,
        deduplicate: bool = True,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Bir nechta query versiyalari bilan qidirish (re-ranking uchun).
        Natijalarni birlashtirib, eng relevantlarini qaytaradi.
        
        Args:
            queries: Savolning turli versiyalari
            top_k: Har bir query uchun natijalar soni
            deduplicate: Takroriy natijalarni olib tashlash
        
        Returns:
            Birlashtirilgan va tartiblangan natijalar
        """
        all_results = {}
        
        for query in queries:
            results = self.search(query, top_k=top_k)
            for chunk, score in results:
                key = (chunk.doc_name, chunk.chunk_index)
                # Maksimal score'ni saqlash
                if key not in all_results or all_results[key][1] < score:
                    all_results[key] = (chunk, score)
        
        # Score bo'yicha tartiblash
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_results[:top_k]
    
    def save(self, save_dir: str) -> None:
        """
        Vector store'ni diskka saqlaydi (keyingi sessiyalarda ishlatish uchun).
        
        Args:
            save_dir: Saqlash papkasi
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS indeksini saqlash
        faiss.write_index(self.index, str(save_path / "faiss.index"))

        # Chunk'larni saqlash
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # Metadata'ni JSON formatda saqlash (o'qish uchun qulay)
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        # Embedding engine'ni saqlash (TF-IDF modeli bilan)
        self.embedding_engine.save(str(save_path / "engine"))

        # Konfiguratsiya
        config = {
            "embedding_model": self.embedding_engine.model_name,
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_engine.embedding_dim,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Vector store saqlandi: {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> "LegalVectorStore":
        """
        Oldin saqlangan vector store'ni yuklaydi.
        
        Args:
            save_dir: Saqlangan vector store papkasi
        
        Returns:
            Tayyor LegalVectorStore
        """
        save_path = Path(save_dir)
        
        if not save_path.exists():
            raise FileNotFoundError(f"Vector store topilmadi: {save_dir}")
        
        # Konfiguratsiyani o'qish
        with open(save_path / "config.json", "r") as f:
            config = json.load(f)
        
        print(f"\n📂 Saqlangan vector store yuklanmoqda: {save_dir}")
        
        # Embedding engine'ni yuklash (TF-IDF modeli bilan)
        engine = EmbeddingEngine.load(str(save_path / "engine"))

        # Store yaratish
        store = cls(engine)
        
        # FAISS indeksini yuklash
        store.index = faiss.read_index(str(save_path / "faiss.index"))
        
        # Chunk'larni yuklash
        with open(save_path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        
        # Metadata
        with open(save_path / "metadata.json", "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        
        store.is_built = True
        
        print(f"   ✅ {len(store.chunks)} ta chunk yuklandi!")
        
        return store
