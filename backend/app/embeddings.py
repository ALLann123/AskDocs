from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from app.config import settings

class EmbeddingGemma(Embeddings):
    """
    Thin wrapper around Ollama's EmbeddingGemma-300M
    """

    def __init__(self)-> None:
        print(f"[EMBEDDINGS] Using {settings.embed_model} via Ollama")
        self._model=OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )

    def embed_documents(self, texts:list[str])-> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingGemma:
    return EmbeddingGemma()