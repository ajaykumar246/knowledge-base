"""
Application Configuration Module.

Loads and validates environment variables from .env using Pydantic.
Provides centralized access to application-wide settings such as Redis,
ChromaDB, Hugging Face embeddings, and OpenAI API credentials.
"""

from functools import lru_cache
import os
from typing import Optional
from dotenv import load_dotenv

# Pre-load environment variables from .env if present
load_dotenv()

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        """Application settings loaded from environment variables and .env file."""

        # Application Metadata
        PROJECT_NAME: str = "RAG-Powered Knowledge Base API"
        VERSION: str = "1.0.0"
        DEBUG: bool = True
        API_V1_PREFIX: str = "/api/v1"
        HOST: str = "0.0.0.0"
        PORT: int = 8000

        # Redis Configuration (Chat History Memory)
        REDIS_URL: str = "redis://localhost:6379/0"
        REDIS_TTL_SECONDS: int = 86400  # Default: 24-hour retention

        # ChromaDB Configuration (Vector Database)
        CHROMA_PERSIST_DIR: str = "./data/chroma_db"
        CHROMA_COLLECTION_NAME: str = "knowledge_base"

        # Hugging Face Embeddings Configuration
        EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
        EMBEDDING_DEVICE: str = "cpu"  # Options: 'cpu' or 'cuda'

        # OpenAI Configuration (LLM & Agent Reasoning)
        OPENAI_API_KEY: Optional[str] = ""
        LLM_MODEL_NAME: str = "gpt-4o-mini"
        LLM_TEMPERATURE: float = 0.0

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=True,
        )

except ImportError:
    # Graceful fallback if pydantic-settings is not yet installed
    from pydantic import BaseModel

    class Settings(BaseModel):  # type: ignore[no-redef]
        """Fallback settings implementation using standard Pydantic and os.environ."""

        PROJECT_NAME: str = os.getenv("PROJECT_NAME", "RAG-Powered Knowledge Base API")
        VERSION: str = os.getenv("VERSION", "1.0.0")
        DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "t")
        API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "8000"))

        REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        REDIS_TTL_SECONDS: int = int(os.getenv("REDIS_TTL_SECONDS", "86400"))

        CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

        EMBEDDING_MODEL_NAME: str = os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
        LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of the application settings.
    Avoids re-reading the .env file on every dependency injection call.
    """
    return Settings()


# Convenient global settings accessor
settings = get_settings()

