"""
Chat and Agentic RAG Controller.

Exposes the POST /chat endpoint which interacts with the LangChain Agent,
checks Redis conversation history, queries ChromaDB when needed via tool calling,
and returns an AI response with source citations.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.chat_agent import execute_agent_chat
from app.services.memory import chat_memory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agentic Chat & Retrieval"])


class ChatRequest(BaseModel):
    """Schema for incoming chat request."""

    session_id: str = Field(
        ...,
        description="Unique session identifier to maintain conversation history in Redis.",
        example="session-dev-123",
        min_length=1,
    )
    message: str = Field(
        ...,
        description="User prompt, question, or conversational message.",
        example="What are the key findings discussed in the uploaded report?",
        min_length=1,
    )


class ChatResponse(BaseModel):
    """Schema for chat response returned to client."""

    session_id: str
    query: str
    answer: str
    tool_calls: List[str] = Field(
        default_factory=list,
        description="List of tools invoked by the LangChain agent during reasoning.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Document sources referenced by the agent to formulate the answer.",
    )


class ClearSessionResponse(BaseModel):
    """Schema for session clearing confirmation."""

    session_id: str
    status: str
    message: str


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Agentic Chat with Memory and Tool Calling",
    description=(
        "Processes user query through an intelligent LangChain agent. "
        "The agent checks Redis for prior conversation memory and decides via tool calling "
        "whether to search ChromaDB documents ('Search_Documents') or answer directly."
    ),
)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Executes conversational agent reasoning over user input and Redis memory.
    """
    try:
        result = await execute_agent_chat(
            session_id=payload.session_id,
            user_message=payload.message,
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Failed to process chat request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating a response: {exc}",
        ) from exc


@router.delete(
    "/chat/{session_id}",
    response_model=ClearSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear Chat Session History",
    description="Purges all cached conversation history for a given session from Redis.",
)
async def clear_session(session_id: str) -> ClearSessionResponse:
    """Clears conversation memory in Redis for the specified session_id."""
    await chat_memory.clear_session(session_id)
    return ClearSessionResponse(
        session_id=session_id,
        status="success",
        message=f"Conversation memory for session '{session_id}' has been cleared.",
    )

