from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    use_rag: bool = True


class SourceDocument(BaseModel):
    source: str
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    used_rag: bool
    sources: list[SourceDocument] = Field(default_factory=list)
