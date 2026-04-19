from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument
from app.services.llm_service import LLMService, get_llm_service
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(tags=["chat"])


@router.get("/rag/status")
async def rag_status(rag_service: RAGService = Depends(get_rag_service)) -> dict[str, int]:
    return {"document_count": rag_service.document_count()}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        context = ""
        docs = []

        if request.use_rag:
            context, docs = rag_service.retrieve_context(request.message)

        answer = llm_service.chat(
            user_message=request.message,
            context=context if context else None,
        )

        sources = [
            SourceDocument(
                source=str(doc.metadata.get("source", "unknown")),
                content_preview=doc.page_content[:120],
            )
            for doc in docs
        ]

        return ChatResponse(
            answer=answer,
            used_rag=bool(docs),
            sources=sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    try:
        context = ""
        if request.use_rag:
            context, _ = rag_service.retrieve_context(request.message)

        def stream_generator() -> Iterator[bytes]:
            try:
                for token in llm_service.stream_chat(
                    user_message=request.message,
                    context=context if context else None,
                ):
                    yield token.encode("utf-8")
            except Exception as exc:  # noqa: BLE001
                yield f"\n[STREAM_ERROR] {exc}".encode("utf-8")

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
