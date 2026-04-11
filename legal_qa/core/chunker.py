"""
=============================================================
LEGAL QA SYSTEM — STEP 2: Smart Text Chunker
=============================================================
Matnni semantik jihatdan to'g'ri bo'laklarga ajratadi.
Overlap yordamida kontekst yo'qolishini oldini oladi.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

from .document_loader import LoadedDocument, DocumentChunk, extract_section_hint


# -------------------------------------------------------
# Chunking Configuration
# -------------------------------------------------------

@dataclass
class ChunkConfig:
    """Chunking parametrlari"""
    
    # Chunk hajmi (taxminiy belgi soni)
    chunk_size: int = 800
    
    # Overlap (ketma-ket chunk'lar o'rtasidagi umumiy belgilar)
    overlap: int = 150
    
    # Minimal chunk hajmi (kichkina chunk'larni o'tkazib yuborish)
    min_chunk_size: int = 100
    
    # Bo'lim/modda bo'yicha ajratishga urinish
    split_on_sections: bool = True


# -------------------------------------------------------
# Smart Splitter
# -------------------------------------------------------

def get_split_points(text: str, split_on_sections: bool = True) -> List[int]:
    """
    Matnni qayerdan bo'lish kerakligini aniqlaydi.
    Ustuvorlik tartibi:
    1. Modda/bo'lim sarlavhalari (Article, Section, ...)
    2. Ikki bo'sh qator (paragraflar orasidagi bo'shliq)
    3. Bir bo'sh qator
    4. Nuqta (.)
    
    Args:
        text: Bo'linadigan matn
        split_on_sections: Bo'lim sarlavhalarida bo'lishni yoqish
    
    Returns:
        Tavsiya etilgan bo'linish nuqtalari (indekslar)
    """
    split_points = []
    
    if split_on_sections:
        # Modda va bo'lim sarlavhalarini topish
        section_patterns = [
            r'\n(?=(?i:article|section|chapter|clause|paragraph)\s+\d)',
            r'\n(?=(?i:moddasi?|bo\'lim)\s+\d)',
            r'\n(?=\d+\.\d*\s+[A-Z])',   # "1.2 Title" kabi raqamlangan sarlavhalar
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                split_points.append(match.start())
    
    # Ikki bo'sh qator
    for match in re.finditer(r'\n\n+', text):
        split_points.append(match.start())
    
    # Bir bo'sh qator
    for match in re.finditer(r'\n', text):
        split_points.append(match.start())
    
    return sorted(set(split_points))


def split_text_with_overlap(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
    min_chunk_size: int = 100,
    split_on_sections: bool = True,
) -> List[Tuple[str, int, int]]:
    """
    Matnni overlap bilan bo'laklarga ajratadi.
    
    Bu funksiya oddiy belgi bo'lishidan ko'ra aqlliroq:
    - Avval tabiiy bo'linish nuqtalarini topadi
    - Keyin chunk_size ga yaqin joyda bo'ladi
    - Overlap uchun oldingi chunk oxirini qo'shadi
    
    Args:
        text: Ajratiladigan matn
        chunk_size: Maqsadli chunk hajmi (belgilar)
        overlap: Qo'shni chunk'lar orasidagi umumiy belgilar
        min_chunk_size: Minimal chunk hajmi
        split_on_sections: Bo'lim sarlavhalarida bo'lish
    
    Returns:
        (chunk_text, char_start, char_end) tuple'lari ro'yxati
    """
    if not text.strip():
        return []
    
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]
    
    # Tabiiy bo'linish nuqtalarini topish
    split_points = get_split_points(text, split_on_sections)
    split_points = [0] + split_points + [len(text)]
    split_points = sorted(set(split_points))
    
    chunks = []
    current_start = 0
    
    while current_start < len(text):
        # Maqsadli tugash nuqtasi
        target_end = current_start + chunk_size
        
        if target_end >= len(text):
            # Oxirgi chunk
            chunk_text = text[current_start:].strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append((chunk_text, current_start, len(text)))
            break
        
        # Maqsadli tugash nuqtasiga eng yaqin tabiiy bo'linish nuqtasini topish
        best_split = target_end
        best_distance = chunk_size  # Maksimal qidiruv masofasi
        
        for point in split_points:
            if current_start < point <= target_end + 200:
                distance = abs(point - target_end)
                if distance < best_distance:
                    best_distance = distance
                    best_split = point
        
        chunk_text = text[current_start:best_split].strip()
        
        if len(chunk_text) >= min_chunk_size:
            chunks.append((chunk_text, current_start, best_split))
        
        # Keyingi chunk uchun boshlanish nuqtasi (overlap bilan)
        current_start = max(current_start + 1, best_split - overlap)
    
    return chunks


# -------------------------------------------------------
# Document → Chunks Pipeline
# -------------------------------------------------------

def create_chunks_from_document(
    document: LoadedDocument,
    config: ChunkConfig = None,
) -> List[DocumentChunk]:
    """
    Yuklangan hujjatdan barcha chunk'larni yaratadi.
    
    Args:
        document: Yuklangan hujjat
        config: Chunking konfiguratsiyasi
    
    Returns:
        DocumentChunk ro'yxati (metadata bilan)
    """
    if config is None:
        config = ChunkConfig()
    
    all_chunks = []
    chunk_index = 0
    
    # PDF uchun sahiha bo'yicha ishlash (sahifa raqamini saqlash)
    if document.file_type == 'pdf':
        for page_num, page_text in enumerate(document.pages, start=1):
            if not page_text.strip():
                continue
            
            text_chunks = split_text_with_overlap(
                page_text,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                min_chunk_size=config.min_chunk_size,
                split_on_sections=config.split_on_sections,
            )
            
            for chunk_text, char_start, char_end in text_chunks:
                section_hint = extract_section_hint(chunk_text)
                
                all_chunks.append(DocumentChunk(
                    text=chunk_text,
                    doc_name=document.name,
                    doc_path=document.path,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    section_hint=section_hint,
                    char_start=char_start,
                    char_end=char_end,
                ))
                chunk_index += 1
    
    else:
        # DOCX va TXT uchun to'liq matn ustida ishlash
        text_chunks = split_text_with_overlap(
            document.full_text,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size,
            split_on_sections=config.split_on_sections,
        )
        
        for chunk_text, char_start, char_end in text_chunks:
            # Taxminiy sahifa raqamini hisoblash (har 2000 belgida 1 sahifa)
            approx_page = (char_start // 2000) + 1
            section_hint = extract_section_hint(chunk_text)
            
            all_chunks.append(DocumentChunk(
                text=chunk_text,
                doc_name=document.name,
                doc_path=document.path,
                page_number=approx_page,
                chunk_index=chunk_index,
                section_hint=section_hint,
                char_start=char_start,
                char_end=char_end,
            ))
            chunk_index += 1
    
    return all_chunks


def create_chunks_from_documents(
    documents: List[LoadedDocument],
    config: ChunkConfig = None,
) -> List[DocumentChunk]:
    """
    Barcha hujjatlardan chunk'lar yaratadi.
    
    Args:
        documents: Yuklangan hujjatlar ro'yxati
        config: Chunking konfiguratsiyasi
    
    Returns:
        Barcha chunk'larning birlashtirilgan ro'yxati
    """
    if config is None:
        config = ChunkConfig()
    
    all_chunks = []
    
    print(f"\n✂️  Matn bo'laklarga ajratilmoqda...")
    print(f"   Chunk hajmi: ~{config.chunk_size} belgi")
    print(f"   Overlap: {config.overlap} belgi")
    print("-" * 50)
    
    for doc in documents:
        chunks = create_chunks_from_document(doc, config)
        all_chunks.extend(chunks)
        print(f"  📄 {doc.name}: {len(chunks)} ta chunk yaratildi")
    
    print(f"\n📊 Jami: {len(all_chunks)} ta chunk")
    
    return all_chunks
