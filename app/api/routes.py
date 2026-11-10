"""
API Routes and Health Check Controller.

Exposes endpoints for:
- Health checking Redis and ChromaDB infrastructure
- System info and runtime configurations
- Placeholder routing for document ingestion (Phase 2) and RAG chat (Phase 3)
"""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import check_chroma_connection, check_redis_connection

router = APIRouter(tags=["System & Health"])


class HealthResponse(BaseModel):
    """Schema for system health response."""

    status: str
    timestamp: str
    environment: str
    services: Dict[str, Any]


class InfoResponse(BaseModel):
    """Schema for application metadata and active configuration."""

    project_name: str
    version: str
    embedding_model: str
    llm_model: str
    chroma_collection: str
    api_prefix: str


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Infrastructure Health Check",
    description="Inspects real-time connectivity status for local ChromaDB and Redis cache.",
)
async def health_check() -> HealthResponse:
    """
    Performs ping checks against Redis and ChromaDB to verify system health.
    """
    redis_health = await check_redis_connection()
    chroma_health = check_chroma_connection()

    # Determine overall system health
    # System is healthy if services respond (or report local readiness)
    overall_status = "healthy"
    if redis_health.get("status") not in ("connected", "unavailable"):
        overall_status = "degraded"
    if chroma_health.get("status") not in ("connected", "unavailable"):
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment="development" if settings.DEBUG else "production",
        services={
            "redis": redis_health,
            "chromadb": chroma_health,
        },
    )


@router.get(
    "/info",
    response_model=InfoResponse,
    status_code=status.HTTP_200_OK,
    summary="System Information",
    description="Returns current project metadata, active models, and configuration details.",
)
async def system_info() -> InfoResponse:
    """Returns active runtime metadata."""
    return InfoResponse(
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        llm_model=settings.LLM_MODEL_NAME,
        chroma_collection=settings.CHROMA_COLLECTION_NAME,
        api_prefix=settings.API_V1_PREFIX,
    )

