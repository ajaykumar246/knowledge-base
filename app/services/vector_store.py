"""
Vector Store Service.

Provides a unified interface to the local ChromaDB vector database
wrapped with LangChain Chroma integrations and Hugging Face embeddings.
"""

import logging
from typing import Any, List, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    # Graceful fallback Document class if langchain-core is not yet installed
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str = "", metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from app.core.config import settings
from app.core.database import get_chroma_client
from app.services.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Global cached LangChain Chroma instance
_vector_store_instance: Optional[Any] = None


def get_vector_store() -> Any:
    """
    Retrieves or initializes the LangChain Chroma vector store instance.
    Uses the local Chroma PersistentClient and Hugging Face embeddings.

    Returns:
        LangChain Chroma vector store instance.
    """
    global _vector_store_instance
    if _vector_store_instance is not None:
        return _vector_store_instance

    client = get_chroma_client()
    embedding_model = get_embedding_model()

    logger.info(
        "Initializing LangChain Chroma vector store (Collection: '%s', Persist Dir: '%s')...",
        settings.CHROMA_COLLECTION_NAME,
        settings.CHROMA_PERSIST_DIR,
    )

    # 1. Try modern langchain_chroma
    try:
        from langchain_chroma import Chroma

        _vector_store_instance = Chroma(
            client=client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
        )
        logger.info("Chroma vector store initialized using langchain_chroma.")
        return _vector_store_instance
    except ImportError:
        pass

    # 2. Try langchain_community fallback
    try:
        from langchain_community.vectorstores import Chroma

        _vector_store_instance = Chroma(
            client=client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
        )
        logger.info("Chroma vector store initialized using langchain_community.")
        return _vector_store_instance
    except Exception as exc:
        logger.error("Failed to initialize Chroma vector store: %s", exc)
        raise RuntimeError(
            f"Could not initialize Chroma vector store: {exc}. "
            "Please ensure 'langchain-chroma' or 'chromadb' is installed."
        ) from exc


def add_documents_to_chroma(documents: List[Document]) -> List[str]:
    """
    Adds a list of LangChain Document objects to the ChromaDB collection.
    Automatically persists embeddings locally.

    Args:
        documents: List of Document chunks with page content and metadata.

    Returns:
        List of generated document IDs in ChromaDB.
    """
    if not documents:
        logger.warning("add_documents_to_chroma called with empty document list.")
        return []

    vector_store = get_vector_store()
    logger.info("Writing %d document chunks to ChromaDB collection...", len(documents))

    ids = vector_store.add_documents(documents)
    logger.info("Successfully indexed %d chunks in ChromaDB.", len(ids))
    return ids


def query_similar_documents(query: str, k: int = 4) -> List[Document]:
    """
    Performs vector similarity search against ChromaDB.

    Args:
        query: User question or search query string.
        k: Number of most similar document chunks to retrieve (default: 4).

    Returns:
        List of retrieved Document objects with content and metadata.
    """
    vector_store = get_vector_store()
    logger.info("Querying ChromaDB for: '%s' (top_k=%d)", query, k)
    results = vector_store.similarity_search(query, k=k)
    logger.info("Retrieved %d relevant chunks from ChromaDB.", len(results))
    return results
