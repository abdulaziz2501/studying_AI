# ⚖️ Legal QA System

**Qattiq (Strict) Huquqiy Savol-Javob Tizimi**

Faqat siz bergan hujjatlar asosida javob beradi. Hech qachon tashqi bilimdan foydalanmaydi.

---

## 🏗️ Loyiha Strukturasi

```
legal_qa/
├── core/
│   ├── document_loader.py   # PDF/DOCX/TXT yuklash
│   ├── chunker.py           # Matnni bo'laklash
│   ├── vector_store.py      # FAISS vektor bazasi
│   └── answering_engine.py  # Qattiq javob berish
├── sample_docs/
│   ├── employment_contract.txt
│   └── service_agreement.txt
├── __init__.py              # build_legal_model() funksiyasi
├── main.py                  # Demo va test
├── requirements.txt
└── README.md
```

---

## 🚀 O'rnatish

```bash
# 1. Kutubxonalarni o'rnatish
pip install -r requirements.txt

# 2. Demo ishga tushirish
python Detecting.py

# 3. O'z hujjatlaringiz bilan
python Detecting.py --folder /path/to/your/documents
```

---

## 💻 Kod bilan Ishlatish

### Oddiy misol:
```python
from legal_qa import build_legal_model

# Tizimni qurish
qa = build_legal_model("./my_legal_docs")

# Savol berish
answer = qa.ask_question("What is the termination notice period?")

# Natijani ko'rish
print(answer.format_display())
```

### Ekran chiqishi:
```
=================================================================
📋  LEGAL QA SYSTEM — JAVOB
=================================================================

❓ SAVOL:
   What is the termination notice period?

-----------------------------------------------------------------

💬 JAVOB [🟢 High ishonch]:

   According to Article 8 of the Employment Contract, either party
   may terminate the contract with 30 (thirty) days written notice.
   [Source: employment_contract.txt, Page 1]

-----------------------------------------------------------------

📚 MANBALAR (3 ta):

   [1] 📄 employment_contract.txt
       📍 Sahifa: 1
       📑 Bo'lim: Article 8
       🔍 Relevantlik: 89.34%
       💬 Parcha: "Either party may terminate this Contract with 30..."

⚠️  DISCLAIMER: This system provides information based ONLY on the
provided legal documents and is NOT legal advice.
=================================================================
```

### API key bilan (aniqroq javoblar):
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

qa = build_legal_model(
    folder_path="./contracts",
    anthropic_api_key="your-api-key",
    chunk_size=800,       # Chunk hajmi (500-1000 tavsiya)
    chunk_overlap=150,    # Overlap (100-200 tavsiya)
    top_k=5,              # Qidiruv natijalari soni
    save_dir="./saved"    # Keyingi safar tez yuklash uchun
)
```

### Saqlangan modelni yuklash:
```python
from legal_qa import load_legal_model

# Birinchi marta emas — tez yuklash
qa = load_legal_model("./saved")
answer = qa.ask_question("penalty for late payment?")
```

### JSON formatda natija:
```python
answer = qa.ask_question("What is the salary?")
import json
print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
```

---

## ⚙️ Sozlamalar

| Parametr | Standart | Tavsiya | Ta'sir |
|----------|---------|---------|--------|
| `chunk_size` | 800 | 500-1000 | Katta = ko'proq kontekst |
| `chunk_overlap` | 150 | 100-200 | Katta = kamroq ma'lumot yo'qolishi |
| `top_k` | 5 | 3-7 | Ko'p = kengroq qidirish |
| `min_similarity_threshold` | 0.35 | 0.3-0.5 | Katta = qattiqroq filtr |

---

## 📊 Embedding Modellari

| Model | Tezlik | Aniqlik | Hajm |
|-------|--------|---------|------|
| `all-MiniLM-L6-v2` | ⚡ Tez | ✅ Yaxshi | 80MB |
| `all-mpnet-base-v2` | 🐢 Sekin | 🏆 Eng yaxshi | 420MB |
| `paraphrase-multilingual-MiniLM-L12-v2` | ⚡ Tez | ✅ Ko'p til | 120MB |

---

## 🔒 Xavfsizlik Kafolatlari

1. **Tashqi bilim ishlatilmaydi** — Faqat yuklangan hujjatlar
2. **Javob topilmasa** — Aniq "Answer not found" qaytaradi
3. **Manba ko'rsatiladi** — Har bir javob uchun hujjat nomi + sahifa
4. **Disclaimer** — Har doim yuridik maslahat emasligi eslatiladi

---

## 🚀 Yaxshilash Takliflari (Step 10)

### 1. Re-ranking (Cross-Encoder)
```python
# Birinchi FAISS bilan 20 ta topib, keyin cross-encoder bilan 5 ta saralash
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

### 2. HyDE (Hypothetical Document Embeddings)
```python
# Savol uchun faraz javob yasab, uni ham embed qilish
hypothetical_answer = llm.generate(f"Write a legal clause about: {question}")
# Keyin shu faraz javobni query sifatida ishlatish
```

### 3. Parent-Child Chunking
```python
# Kichik chunk'larni topib, ularning katta "parent" ni qaytarish
# Aniqlik + kontekst balansini yaxshilaydi
```

### 4. Hybrid Search (BM25 + FAISS)
```python
# Leksik (kalit so'z) + semantik qidiruvni birlashtirish
from rank_bm25 import BM25Okapi
# Weighted combination
```
