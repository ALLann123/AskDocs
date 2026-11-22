from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            temperature=0.2,
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )