from functools import lru_cache
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings

SYSTEM_PROMPT = (
    "你是一个专业、真诚的 AI 对话助手。"
    "请优先基于给定知识上下文回答；如果上下文不足，明确告知并给出稳妥建议。"
)


class LLMService:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure your .env file.")

        kwargs = {
            "api_key": settings.openai_api_key,
            "model": settings.openai_model,
            "temperature": settings.temperature,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        self._llm = ChatOpenAI(**kwargs)

    def _build_messages(self, user_message: str, context: str | None = None) -> list:
        context_block = context.strip() if context else "暂无可用知识库内容。"
        user_prompt = (
            f"用户问题:\n{user_message}\n\n"
            f"可参考知识:\n{context_block}\n\n"
            "请给出清晰、可执行的回答。"
        )
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

    @staticmethod
    def _chunk_to_text(content: object) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        return ""

    def chat(self, user_message: str, context: str | None = None) -> str:
        response = self._llm.invoke(self._build_messages(user_message, context))

        if isinstance(response.content, str):
            return response.content
        return str(response.content)

    def stream_chat(self, user_message: str, context: str | None = None) -> Iterator[str]:
        for chunk in self._llm.stream(self._build_messages(user_message, context)):
            text = self._chunk_to_text(getattr(chunk, "content", ""))
            if text:
                yield text


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService(get_settings())
