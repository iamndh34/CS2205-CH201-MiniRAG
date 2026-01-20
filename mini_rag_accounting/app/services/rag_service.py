import re
from .rag_coa import RagCOA


class RagRouter:
    """
    Router đơn giản: Extract số TK → Tra cứu
    Format câu hỏi: "Tài khoản XXX là gì?"
    """

    @classmethod
    def _extract_account_code(cls, question: str) -> str:
        """Extract mã tài khoản từ câu hỏi"""
        match = re.search(r'\b(\d{3,4})\b', question)
        return match.group(1) if match else None

    @classmethod
    def ask(cls, question: str):
        """
        Hỏi đáp tài khoản - extract số TK và tra cứu
        """
        code = cls._extract_account_code(question)
        print(f"[Router] Question: {question}, Code: {code}")

        if not code:
            yield "Vui lòng nhập số tài khoản cần tra cứu.\nVí dụ: Tài khoản 156 là gì?"
            return

        for chunk in RagCOA.ask(code):
            yield chunk
