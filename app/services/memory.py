"""
Session Chat Memory Service.

Manages conversational history stored in Redis.
Enables the LLM agent to maintain context across multi-turn user interactions,
with configurable TTL expiration to prevent memory leakage.
"""

import json
import logging
from typing import Any, List, Optional

from app.core.config import settings
from app.core.database import get_redis_client

logger = logging.getLogger(__name__)

# Attempt importing LangChain message schemas
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:
    # Graceful fallback dataclasses if langchain-core is not yet installed
    class BaseMessage:  # type: ignore[no-redef]
        def __init__(self, content: str):
            self.content = content

    class HumanMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    class AIMessage(BaseMessage):  # type: ignore[no-redef]
        pass


# In-memory session store fallback if Redis is unavailable
_IN_MEMORY_SESSIONS: dict[str, List[dict[str, str]]] = {}


class RedisSessionMemory:
    """
    Manages session-based chat message history stored in Redis.
    Provides persistence, TTL expiration, and conversion to LangChain message formats.
    """

    def __init__(self, ttl_seconds: Optional[int] = None):
        self.ttl_seconds = ttl_seconds or settings.REDIS_TTL_SECONDS

    def _get_key(self, session_id: str) -> str:
        """Constructs the Redis key for a given session."""
        return f"chat_history:{session_id}"

    async def get_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Retrieves all past conversation messages for a session formatted for LangChain.

        Args:
            session_id: Unique identifier for the user session.

        Returns:
            List of HumanMessage and AIMessage objects.
        """
        key = self._get_key(session_id)
        redis_client = await get_redis_client()

        raw_messages: List[dict] = []

        if redis_client is not None:
            try:
                # Retrieve all message items from Redis list
                items = await redis_client.lrange(key, 0, -1)
                for item in items:
                    try:
                        raw_messages.append(json.loads(item))
                    except json.JSONDecodeError:
                        continue
            except Exception as exc:
                logger.warning(
                    "Error reading session '%s' from Redis: %s. Falling back to memory.",
                    session_id,
                    exc,
                )
                raw_messages = _IN_MEMORY_SESSIONS.get(session_id, [])
        else:
            raw_messages = _IN_MEMORY_SESSIONS.get(session_id, [])

        # Convert serialized records to LangChain BaseMessage objects
        messages: List[BaseMessage] = []
        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))

        logger.debug("Loaded %d historical messages for session '%s'.", len(messages), session_id)
        return messages

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Appends a new message to the session history in Redis and refreshes the TTL.

        Args:
            session_id: Unique identifier for the user session.
            role: 'human' or 'ai'.
            content: Text content of the message.
        """
        key = self._get_key(session_id)
        payload = json.dumps({"role": role, "content": content})
        redis_client = await get_redis_client()

        if redis_client is not None:
            try:
                # Push message to end of Redis list
                await redis_client.rpush(key, payload)
                # Refresh session expiration TTL
                await redis_client.expire(key, self.ttl_seconds)
                logger.debug("Persisted %s message to Redis for session '%s'.", role, session_id)
                return
            except Exception as exc:
                logger.warning(
                    "Error persisting %s message to Redis: %s. Storing in-memory.",
                    role,
                    exc,
                )

        # In-memory fallback
        if session_id not in _IN_MEMORY_SESSIONS:
            _IN_MEMORY_SESSIONS[session_id] = []
        _IN_MEMORY_SESSIONS[session_id].append({"role": role, "content": content})

    async def add_user_message(self, session_id: str, content: str) -> None:
        """Convenience helper to append a user message."""
        await self.add_message(session_id, role="human", content=content)

    async def add_ai_message(self, session_id: str, content: str) -> None:
        """Convenience helper to append an AI response."""
        await self.add_message(session_id, role="ai", content=content)

    async def clear_session(self, session_id: str) -> None:
        """Deletes all conversation history for a given session."""
        key = self._get_key(session_id)
        redis_client = await get_redis_client()
        if redis_client is not None:
            try:
                await redis_client.delete(key)
                logger.info("Cleared session history from Redis for '%s'.", session_id)
            except Exception as exc:
                logger.warning("Error deleting session '%s' from Redis: %s", session_id, exc)

        _IN_MEMORY_SESSIONS.pop(session_id, None)


# Singleton memory instance
chat_memory = RedisSessionMemory()

