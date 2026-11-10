"""
Database and Cache Infrastructure.

Manages connections and health checks for:
1. Redis: In-memory store for session-based chat history.
2. ChromaDB: Local persistent vector store for document embeddings.
"""

import logging
import os
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# ==============================================================================
# Redis Client Connection & Health Management
# ==============================================================================

# Global reference to the async Redis client instance
_async_redis_client: Optional[Any] = None


async def get_redis_client() -> Optional[Any]:
    """
    Retrieves or initializes the async Redis client singleton.
    Uses connection pooling provided by redis.asyncio.

    Returns:
        redis.asyncio.Redis instance if available, otherwise None.
    """
    global _async_redis_client
    if _async_redis_client is not None:
        return _async_redis_client

    try:
        import redis.asyncio as aioredis  # Lazy import for flexibility

        _async_redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        logger.info("Initialized Redis client with URL: %s", settings.REDIS_URL)
        return _async_redis_client
    except ImportError:
        logger.warning("The 'redis' package is not installed. Redis memory will be disabled.")
        return None
    except Exception as exc:
        logger.error("Failed to initialize Redis client: %s", exc)
        return None


async def close_redis_client() -> None:
    """Closes the Redis client connection pool on application shutdown."""
    global _async_redis_client
    if _async_redis_client is not None:
        try:
            await _async_redis_client.aclose()
            logger.info("Redis client connection pool closed successfully.")
        except Exception as exc:
            logger.error("Error while closing Redis client: %s", exc)
        finally:
            _async_redis_client = None


async def check_redis_connection() -> Dict[str, Any]:
    """
    Performs a ping health check on the configured Redis instance.

    Returns:
        Dict containing status ('connected' or 'disconnected') and details.
    """
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return {
            "status": "unavailable",
            "message": "redis package not installed",
        }

    try:
        client = await get_redis_client()
        if client is None:
            return {
                "status": "disconnected",
                "message": "Redis client initialization returned None",
            }
        # Ping the Redis server
        await client.ping()
        return {
            "status": "connected",
            "url": settings.REDIS_URL,
        }
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return {
            "status": "disconnected",
            "error": str(exc),
            "url": settings.REDIS_URL,
        }


# ==============================================================================
# ChromaDB Client Connection & Health Management
# ==============================================================================

# Global reference to the local ChromaDB persistent client
_chroma_client: Optional[Any] = None


def get_chroma_client() -> Optional[Any]:
    """
    Retrieves or initializes the ChromaDB PersistentClient singleton.
    Stores vector embeddings on disk at settings.CHROMA_PERSIST_DIR.

    Returns:
        chromadb.PersistentClient instance if available, otherwise None.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # Ensure persistence directory exists
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
            ),
        )
        logger.info(
            "Initialized local ChromaDB PersistentClient at directory: %s",
            settings.CHROMA_PERSIST_DIR,
        )
        return _chroma_client
    except ImportError:
        logger.warning("The 'chromadb' package is not installed. Vector store will be disabled.")
        return None
    except Exception as exc:
        logger.error("Failed to initialize ChromaDB client: %s", exc)
        return None


def get_or_create_collection(collection_name: Optional[str] = None) -> Optional[Any]:
    """
    Helper to retrieve or create a ChromaDB collection by name.

    Args:
        collection_name: Target collection name (defaults to settings.CHROMA_COLLECTION_NAME).

    Returns:
        Chroma collection instance, or None if ChromaDB is unavailable.
    """
    client = get_chroma_client()
    if client is None:
        return None

    target_name = collection_name or settings.CHROMA_COLLECTION_NAME
    try:
        collection = client.get_or_create_collection(name=target_name)
        return collection
    except Exception as exc:
        logger.error("Error creating/retrieving ChromaDB collection '%s': %s", target_name, exc)
        return None


def check_chroma_connection() -> Dict[str, Any]:
    """
    Performs a heartbeat and directory check on the local ChromaDB instance.

    Returns:
        Dict containing status ('connected' or 'unavailable') and collection metrics.
    """
    try:
        import chromadb
    except ImportError:
        return {
            "status": "unavailable",
            "message": "chromadb package not installed",
        }

    try:
        client = get_chroma_client()
        if client is None:
            return {
                "status": "unavailable",
                "message": "ChromaDB client initialization returned None",
            }

        # Check heartbeat and list collections
        heartbeat = client.heartbeat()
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        return {
            "status": "connected",
            "type": "PersistentClient (Local)",
            "persist_directory": settings.CHROMA_PERSIST_DIR,
            "heartbeat": heartbeat,
            "collections_count": len(collections),
            "collections": collection_names,
        }
    except Exception as exc:
        logger.warning("ChromaDB health check failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "persist_directory": settings.CHROMA_PERSIST_DIR,
        }

