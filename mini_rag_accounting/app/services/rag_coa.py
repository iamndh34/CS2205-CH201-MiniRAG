import json
import os
import time
from typing import Optional

import ollama

from app.core.config import settings
from .stream_utils import stream_by_sentence
from .prompt_template import SYSTEM_PROMPT, build_user_prompt, build_minirag_context

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COA_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_99.json")

if not os.path.exists(COA_FILE):
    print(f"[WARN] {COA_FILE} not found.")
    COA_DATA = []
    COA_FILE_SIZE = 0
else:
    with open(COA_FILE, "r", encoding="utf-8") as f:
        COA_DATA = json.load(f)
    COA_FILE_SIZE = os.path.getsize(COA_FILE)

COA_BY_CODE = {acc["code"]: acc for acc in COA_DATA}

MINIRAG_RESOURCE_INFO = {
    "index_type": "JSON Hashmap",
    "index_size_kb": round(COA_FILE_SIZE / 1024, 2),
    "embedding_model": "Không cần",
    "embedding_model_size_mb": 0,
    "vector_storage_mb": 0,
    "total_documents": len(COA_DATA),
    "lookup_complexity": "O(1)"
}


class RagCOA:
    @classmethod
    def lookup(cls, code: str) -> Optional[dict]:
        return COA_BY_CODE.get(code.strip())

    @classmethod
    def get_resource_info(cls) -> dict:
        """Return resource usage info for MiniRAG"""
        return MINIRAG_RESOURCE_INFO

    @classmethod
    def ask(cls, code: str):
        print(f"\n[MiniRAG] Looking up: {code}")

        retrieval_start = time.perf_counter()
        acc = cls.lookup(code)
        lookup_time = time.perf_counter() - retrieval_start

        if not acc:
            yield f"Không tìm thấy tài khoản {code} trong danh mục Thông tư 99/2025."
            yield f"__RETRIEVAL__:{json.dumps({'time_ms': round(lookup_time * 1000, 3), 'found': False})}"
            return

        retrieval_time = time.perf_counter() - retrieval_start

        context = build_minirag_context(acc)
        print(f"[MiniRAG] Found: TK {acc['code']} - {acc['name']} in {retrieval_time*1000:.3f}ms")

        retrieval_info = {
            "time_ms": round(retrieval_time * 1000, 3),
            "found": True,
            "doc_count": 1,
            "confidence": "100%",
            "path": f"COA_BY_CODE[{code}] -> TK {acc['code']}"
        }
        yield f"__RETRIEVAL__:{json.dumps(retrieval_info, ensure_ascii=False)}"

        # ===== GENERATION PHASE =====
        question = f"Tài khoản {code} là gì?"
        user_prompt = build_user_prompt(question, context)

        generation_start = time.perf_counter()
        slm_output = ""
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            stream = client.chat(
                model=settings.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            for sentence in stream_by_sentence(stream):
                slm_output += sentence
                yield sentence
        except Exception as e:
            print(f"[MiniRAG Error] {e}")

        generation_time = time.perf_counter() - generation_start

        # Fallback nếu SLM không trả về đúng format
        if not slm_output or "1." not in slm_output:
            print("[MiniRAG] Using fallback...")
            yield cls._fallback(acc)
        else:
            yield "\n\n(Căn cứ: Thông tư 99/2025 - MiniRAG)"

        # Send generation time
        yield f"__GENERATION__:{json.dumps({'time_ms': round(generation_time * 1000, 3)})}"

    @classmethod
    def _fallback(cls, acc: dict) -> str:
        """Fallback khi SLM không hoạt động"""
        type_id = acc.get('type_id', 0)
        balance_side = 'Nợ' if type_id in [1, 5] else 'Có' if type_id in [2, 3, 4] else 'Không có số dư cuối kỳ'

        return f"""
1. THÔNG TIN CƠ BẢN:
- Số hiệu: {acc['code']}
- Tên tài khoản: {acc['name']}
- Tên tiếng Anh: {acc.get('name_en', 'N/A')}
- Phân loại: {acc['type_name']}

2. NỘI DUNG PHẢN ÁNH:
Tài khoản {acc['code']} được sử dụng để theo dõi và phản ánh tình hình biến động của {acc['name'].lower()} tại doanh nghiệp.

3. KẾT CẤU:
- Bên Nợ: Ghi nhận phát sinh tăng.
- Bên Có: Ghi nhận phát sinh giảm.
- Số dư: Thường nằm bên {balance_side}.

(Căn cứ: Thông tư 99/2025)"""
