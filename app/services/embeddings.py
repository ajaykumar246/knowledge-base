"""
Embeddings Service.

Initializes and caches the local Hugging Face embedding model.
Uses sentence-transformers models (e.g., 'all-MiniLM-L6-v2') to produce
dense vector representations locally without relying on external API calls.
"""

from functools import lru_cache
import logging
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global singleton reference for embeddings instance
_embedding_instance: Optional[Any] = None


def get_embedding_model() -> Any:
    """
    Returns a cached instance of the local Hugging Face Embeddings model.
    Utilizes langchain_huggingface or langchain_community with fallback.

    Returns:
        Embeddings instance compatible with LangChain vector stores.
    """
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    logger.info(
        "Loading Hugging Face embedding model: %s on device: %s",
        settings.EMBEDDING_MODEL_NAME,
        settings.EMBEDDING_DEVICE,
    )

    # 1. Try langchain_huggingface (LangChain v0.2+ standard)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Successfully loaded embeddings using langchain_huggingface.")
        return _embedding_instance
    except ImportError:
        pass

    # 2. Try langchain_community fallback
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Successfully loaded embeddings using langchain_community.")
        return _embedding_instance
    except ImportError:
        pass

    # 3. Graceful fallback directly via sentence_transformers
    try:
        from sentence_transformers import SentenceTransformer

        class SentenceTransformerEmbeddingsWrapper:
            """Simple LangChain-compatible wrapper around SentenceTransformer."""

            def __init__(self, model_name: str, device: str = "cpu"):
                self.model = SentenceTransformer(model_name, device=device)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                return embeddings.tolist()

            def embed_query(self, text: str) -> list[float]:
                embedding = self.model.encode([text], normalize_embeddings=True)
                return embedding[0].tolist()

        _embedding_instance = SentenceTransformerEmbeddingsWrapper(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )
        logger.info("Loaded embeddings using direct SentenceTransformer wrapper.")
        return _embedding_instance
    except Exception as exc:
        logger.error("Failed to load local embedding model: %s", exc)
        raise RuntimeError(
            f"Could not load Hugging Face embeddings model '{settings.EMBEDDING_MODEL_NAME}'. "
            "Please ensure 'sentence-transformers' or 'langchain-huggingface' is installed."
        ) from exc

