"""
=============================================================
LEGAL QA SYSTEM — Demo va Test Script
=============================================================
Ishga tushirish:
    python Detecting.py
    python Detecting.py --folder /path/to/your/docs
    python Detecting.py --save ./my_model --load ./my_model

Bu script:
1. Sample hujjatlar bilan tizimni quradi
2. Bir nechta test savollar beradi
3. Natijalarni ekranda ko'rsatadi
"""

import sys
import os
import argparse

# Loyiha papkasini (ya'ni studying_AI) Python path ga qo'shish
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legal_qa import build_legal_model, load_legal_model


# -------------------------------------------------------
# Test Savollar
# -------------------------------------------------------

TEST_QUESTIONS = [
    # ✅ Javob topilishi kerak
    "What is the monthly salary for the employee?",
    "How many days of annual leave does the employee get?",
    "What is the notice period for contract termination?",
    "What are the payment terms for late invoices?",
    "How long is the warranty period for delivered software?",
    
    # 🔍 Murakkabroq savollar
    "What happens if the service provider delays delivery?",
    "What are the confidentiality obligations after termination?",
    "Can the employee work for competitors after leaving?",
    
    # ❌ Javob topilmasligi kerak
    "What is the company's office address?",
    "What is the CEO's personal email address?",
]


# -------------------------------------------------------
# Main Runner
# -------------------------------------------------------

def run_demo(
    folder_path: str,
    api_key: str = None,
    save_dir: str = None,
    load_dir: str = None,
):
    """
    Demo ishga tushirish funksiyasi.
    
    Args:
        folder_path: Hujjatlar papkasi
        api_key: Anthropic API key (ixtiyoriy)
        save_dir: Saqlash papkasi (ixtiyoriy)
        load_dir: Yuklash papkasi (ixtiyoriy)
    """
    print("\n" + "🔷" * 30)
    print("⚖️  LEGAL QA SYSTEM — DEMO")
    print("🔷" * 30)
    
    # ── Tizimni yuklash yoki qurish ──────────────────────────
    
    if load_dir and os.path.exists(load_dir):
        print(f"\n⚡ Saqlangan model yuklanmoqda: {load_dir}")
        qa = load_legal_model(load_dir, anthropic_api_key=api_key)
    else:
        print(f"\n🔨 Model qurilmoqda: {folder_path}")
        qa = build_legal_model(
            folder_path=folder_path,
            anthropic_api_key=api_key,
            chunk_size=700,
            chunk_overlap=150,
            top_k=4,
            save_dir=save_dir,
        )
    
    # ── Interaktiv rejim yoki test rejim ─────────────────────
    
    mode = input("\n\n📌 Rejimni tanlang:\n   [1] Avtomat test savollar\n   [2] O'zingiz savol bering\n\n   Tanlov (1/2): ").strip()
    
    if mode == "2":
        interactive_mode(qa)
    else:
        auto_test_mode(qa)


def auto_test_mode(qa):
    """Avtomatik test savollarni ishga tushiradi"""
    
    print("\n\n" + "=" * 65)
    print("🤖 AVTOMATIK TEST REJIMI")
    print("=" * 65)
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n\n[SAVOL {i}/{len(TEST_QUESTIONS)}]")
        answer = qa.ask_and_print(question)
        
        # Qisqa pauza
        input("\n  ⏎ Davom etish uchun Enter bosing...")
    
    print("\n\n✅ Barcha test savollar javoblandi!")


def interactive_mode(qa):
    """Foydalanuvchi o'z savolini beradi"""
    
    print("\n\n" + "=" * 65)
    print("💬 INTERAKTIV REJIM")
    print("   Chiqish uchun 'exit' yoki 'q' yozing")
    print("=" * 65)
    
    while True:
        print("\n")
        question = input("❓ Savolingiz: ").strip()
        
        if question.lower() in ('exit', 'q', 'quit', 'chiq'):
            print("\n👋 Dastur yopildi.")
            break
        
        if not question:
            print("   ⚠️  Bo'sh savol. Qayta urinib ko'ring.")
            continue
        
        answer = qa.ask_and_print(question)
        
        # JSON formatda ko'rish imkoniyati
        show_json = input("\n   📋 JSON formatda ko'rish? (j/n): ").strip().lower()
        if show_json == 'j':
            import json
            print("\n" + json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal QA System — Huquqiy savol-javob tizimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Misollar:
  python Detecting.py                                   # Demo hujjatlar bilan
  python Detecting.py --folder /path/to/docs            # O'z hujjatlaringiz
  python Detecting.py --folder ./docs --save ./model    # Qurib saqlash
  python Detecting.py --load ./model                    # Saqlangan modelni yuklash
  python Detecting.py --folder ./docs --api YOUR_KEY    # Claude API bilan
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default='legal_qa/sample_docs',
        help="Hujjatlar papkasi (standart: sample_docs)"
    )
    parser.add_argument(
        '--api',
        type=str,
        default=None,
        help="Anthropic API key (yoki ANTHROPIC_API_KEY env variable)"
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help="Tizimni saqlash papkasi"
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help="Saqlangan tizimni yuklash"
    )
    
    args = parser.parse_args()
    
    # API keyni environment variable dan ham olish mumkin
    api_key = args.api or os.environ.get("ANTHROPIC_API_KEY")
    
    run_demo(
        folder_path=args.folder,
        api_key=api_key,
        save_dir=args.save,
        load_dir=args.load,
    )
