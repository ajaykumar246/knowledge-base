"""
Main FastAPI Application Entrypoint.

Initializes the FastAPI application, registers middleware, configures lifespan events,
and mounts API versioned routes.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.routes import router as system_router
from app.api.upload import router as upload_router
from app.core.config import settings
from app.core.database import close_redis_client, get_chroma_client, get_redis_client

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_knowledge_base")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application Lifespan Context Manager.
    Handles startup infrastructure initialization and graceful resource shutdown.
    """
    logger.info("Starting up %s (v%s)...", settings.PROJECT_NAME, settings.VERSION)

    # 1. Ensure local storage directories exist
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs("./uploads", exist_ok=True)
    logger.info("Storage directories ensured: %s, ./uploads", settings.CHROMA_PERSIST_DIR)

    # 2. Warm up database connections (non-blocking)
    try:
        get_chroma_client()
        logger.info("ChromaDB persistent client initialized.")
    except Exception as exc:
        logger.warning("ChromaDB initialization deferred: %s", exc)

    try:
        await get_redis_client()
        logger.info("Redis client initialized.")
    except Exception as exc:
        logger.warning("Redis client initialization deferred: %s", exc)

    yield  # Application is now running and serving requests

    # 3. Clean up connections on shutdown
    logger.info("Shutting down %s...", settings.PROJECT_NAME)
    await close_redis_client()
    logger.info("Cleanup complete. Goodbye!")


# Create FastAPI instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-ready RAG (Retrieval-Augmented Generation) Knowledge Base API "
        "built with FastAPI, LangChain, local ChromaDB, Redis chat history, "
        "and Hugging Face local embeddings."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
# Mounting both under versioned prefix (/api/v1) and top-level convenience alias
app.include_router(system_router, prefix=settings.API_V1_PREFIX)
app.include_router(upload_router, prefix=settings.API_V1_PREFIX)
app.include_router(upload_router)  # Direct /upload convenience alias
app.include_router(chat_router, prefix=settings.API_V1_PREFIX)
app.include_router(chat_router)  # Direct /chat convenience alias


@app.get(
    "/",
    tags=["Root"],
    summary="Root Welcome Endpoint",
    description="Returns a welcome message, API metadata, and quick links to interactive documentation.",
)
async def root() -> JSONResponse:
    """Root endpoint welcoming users and providing navigation links."""
    return JSONResponse(
        content={
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "online",
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json",
            },
            "endpoints": {
                "health": f"{settings.API_V1_PREFIX}/health",
                "info": f"{settings.API_V1_PREFIX}/info",
                "upload": "/upload",
                "chat": "/chat",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

