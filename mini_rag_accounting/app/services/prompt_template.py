SYSTEM_PROMPT = """Bạn là trợ lý kế toán Việt Nam chuyên nghiệp.
Dựa trên thông tin được cung cấp, hãy trả lời câu hỏi một cách chi tiết và chính xác.
LUÔN trả lời bằng TIẾNG VIỆT.
Trả lời dựa trên Thông tư 99/2025."""

OUTPUT_FORMAT = """
1. THÔNG TIN CƠ BẢN:
- Số hiệu: ...
- Tên tài khoản: ...
- Tên tiếng Anh: ...
- Loại tài khoản: ...

2. NỘI DUNG PHẢN ÁNH:
...

3. KẾT CẤU VÀ NỘI DUNG:
- Bên Nợ: ...
- Bên Có: ...
- Số dư: ...
"""


def build_user_prompt(question: str, context: str) -> str:
    return f"""Câu hỏi: {question}

Thông tin tài khoản theo Thông tư 99/2025:
{context}

Hãy trả lời câu hỏi dựa trên thông tin trên.

Trả lời theo định dạng:{OUTPUT_FORMAT}"""


def build_minirag_context(acc: dict) -> str:
    return f"""Tài khoản: {acc['code']}
- Tên: {acc['name']}
- Tên tiếng Anh: {acc.get('name_en', 'N/A')}
- Loại: {acc['type_name']}"""


def build_traditional_context(results: list) -> str:
    context_parts = []
    for code, score, acc in results:
        context_parts.append(f"""Tài khoản: {acc['code']}
- Tên: {acc['name']}
- Tên tiếng Anh: {acc.get('name_en', 'N/A')}
- Loại: {acc['type_name']}
- Độ tương đồng: {score:.2%}""")
    return "\n\n".join(context_parts)
