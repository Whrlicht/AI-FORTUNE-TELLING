from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="FastAPI + LangChain + RAG initial scaffold",
    )

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(chat_router, prefix="/api/v1")
    return app


app = create_app()
