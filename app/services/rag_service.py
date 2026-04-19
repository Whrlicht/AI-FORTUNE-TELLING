from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings, get_settings


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        persist_dir = Path(settings.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
        self._vector_store = Chroma(
            collection_name="knowledge_base",
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    def retrieve_context(self, query: str) -> tuple[str, list[Document]]:
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": self._settings.retrieval_k}
        )
        docs = retriever.invoke(query)

        if not docs:
            return "", []

        context = "\n\n".join(
            [f"[文档{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)]
        )
        return context, docs

    def document_count(self) -> int:
        return self._vector_store._collection.count()  # noqa: SLF001

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        if not texts:
            return

        prepared_metadatas = metadatas if metadatas else [{} for _ in texts]
        self._vector_store.add_texts(texts=texts, metadatas=prepared_metadatas)


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(get_settings())
