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

def get_section_split_points(text: str) -> List[int]:
    """
    Faqat modda va bo'lim sarlavhalari bo'yicha bo'linish nuqtalarini topadi.
    """
    split_points = []
    
    # Modda va bo'lim sarlavhalarini ushlash uchun qoidalar
    section_patterns = [
        r'\n(?=(?i:article|section|chapter|clause|paragraph)\s+\d)',
        r'\n(?=(?i:moddasi?|moddalari)\s+\d)',  # masalan: "moddasi 12", "moddalari"
        r'\n(?=(?i:bo\'lim)\s+\d)',
        r'\n(?=\d+-modda)',                     # masalan: "12-modda"
        r'\n(?=\d+\.\s+(?i:modda|bo\'lim))',    # masalan: "1. Modda"
        r'\n(?=\d+\.\d*\s+[A-Z])',              # "1.2 Title"
    ]
    
    for pattern in section_patterns:
        for match in re.finditer(pattern, text):
            split_points.append(match.start())
            
    # Hujjat boshidagi sarlavhalar uchun (agar \n bilan boshlanmasa)
    start_patterns = [
        r'^(?=(?i:article|section|chapter|clause|paragraph)\s+\d)',
        r'^(?=(?i:moddasi?|moddalari)\s+\d)',
        r'^(?=(?i:bo\'lim)\s+\d)',
        r'^\d+-modda',
        r'^\d+\.\s+(?i:modda|bo\'lim)',
    ]
    
    for pattern in start_patterns:
        if re.search(pattern, text):
            split_points.append(0)
            break
            
    return sorted(set(split_points))


def get_paragraph_split_points(text: str) -> List[int]:
    """Oddiy paragraf va \n orqali bo'linish nuqtalari (moddalar topilmasa)"""
    split_points = []
    for match in re.finditer(r'\n\n+', text):
        split_points.append(match.start())
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
    Matnni moddalar (articles) yoki overlap bilan bo'laklarga ajratadi.
    Agar matnda moddalar ko'rsatilgan bo'lsa, har bir modda bitta yaxlit chunk bo'ladi.
    """
    if not text.strip():
        return []
        
    chunks = []
    
    if split_on_sections:
        # Moddalar bo'yicha ajratishga urinamiz
        section_points = get_section_split_points(text)
        
        if len(section_points) > 0:
            if 0 not in section_points:
                section_points.insert(0, 0)
            if len(text) not in section_points:
                section_points.append(len(text))
                
            for i in range(len(section_points) - 1):
                start = section_points[i]
                end = section_points[i + 1]
                
                chunk_text = text[start:end].strip()
                if len(chunk_text) >= min_chunk_size:
                    chunks.append((chunk_text, start, end))
                    
            if chunks:
                return chunks
                
    # Moddalar topilmasa yoki split_on_sections=False bo'lsa, uzunligi bo'yicha overlap qilib qirqamiz
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]
        
    split_points = get_paragraph_split_points(text)
    split_points = [0] + split_points + [len(text)]
    split_points = sorted(set(split_points))
    
    current_start = 0
    while current_start < len(text):
        target_end = current_start + chunk_size
        
        if target_end >= len(text):
            chunk_text = text[current_start:].strip()
            if len(chunk_text) >= min_chunk_size:
                chunks.append((chunk_text, current_start, len(text)))
            break
            
        best_split = target_end
        best_distance = chunk_size
        
        for point in split_points:
            if current_start < point <= target_end + 200:
                distance = abs(point - target_end)
                if distance < best_distance:
                    best_distance = distance
                    best_split = point
                    
        chunk_text = text[current_start:best_split].strip()
        if len(chunk_text) >= min_chunk_size:
            chunks.append((chunk_text, current_start, best_split))
            
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
