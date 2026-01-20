from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
from app.services.rag_service import RagRouter
from app.services.rag_traditional import RagTraditional
from app.services.rag_coa import RagCOA


router = APIRouter(
    prefix="/accounting",
    tags=["Accounting AI"]
)


@router.get("/resource-info")
async def get_resource_info():
    """
    Get resource usage info for both RAG methods.
    Used to display resource comparison in UI.
    """
    return JSONResponse({
        "minirag": RagCOA.get_resource_info(),
        "traditional": RagTraditional.get_resource_info()
    })


@router.get("/ask")
async def ask_accounting(
    question: str = Query(..., min_length=2, description="Câu hỏi về kế toán")
):
    """
    MiniRAG: Tra cứu tài khoản theo Thông tư 99/2025
    - Direct lookup by code
    - SLM generation
    """
    return StreamingResponse(
        RagRouter.ask(question),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/ask-traditional")
async def ask_traditional(
    question: str = Query(..., min_length=2, description="Câu hỏi về kế toán")
):
    """
    Traditional RAG: Embedding search + LLM
    - Vector similarity search
    - Full context generation
    """
    return StreamingResponse(
        RagTraditional.ask(question),
        media_type="text/plain; charset=utf-8"
    )
