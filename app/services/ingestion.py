"""
Asynchronous Ingestion Pipeline.

Handles background processing of uploaded PDF documents:
1. Loading PDF pages via LangChain's PyPDFLoader
2. Chunking text with RecursiveCharacterTextSplitter
3. Attaching metadata (source, chunk index, timestamp, task id)
4. Generating local Hugging Face embeddings and storing chunks in ChromaDB
5. Safely cleaning up temporary upload files
"""

from datetime import datetime, timezone
import logging
import os
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    # Graceful fallback Document class if langchain-core is not yet installed
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str = "", metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from app.services.vector_store import add_documents_to_chroma

logger = logging.getLogger(__name__)


def _get_text_splitter():
    """Returns an instance of RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def process_pdf_background(
    file_path: str,
    original_filename: str,
    task_id: str,
    delete_temp_file: bool = True,
) -> None:
    """
    Background worker executed via FastAPI BackgroundTasks.

    Parses the uploaded PDF file, generates text chunks, creates embeddings,
    and indexes them into the local ChromaDB vector store without blocking
    the HTTP request thread.

    Args:
        file_path: Local filesystem path to the temporarily saved PDF file.
        original_filename: Original name of the uploaded PDF file.
        task_id: Unique UUID string tracking the background task.
        delete_temp_file: Whether to delete the temporary file after ingestion.
    """
    logger.info(
        "[Task %s] Starting asynchronous ingestion for file: '%s' (Path: %s)",
        task_id,
        original_filename,
        file_path,
    )

    start_time = datetime.now(timezone.utc)

    try:
        # Step 1: Verify file existence
        if not os.path.exists(file_path):
            logger.error("[Task %s] File not found at path: %s", task_id, file_path)
            return

        # Step 2: Load PDF pages with LangChain PyPDFLoader
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as exc:
            logger.error("[Task %s] pypdf / langchain-community not available: %s", task_id, exc)
            return

        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        if not raw_documents:
            logger.warning(
                "[Task %s] No text content extracted from '%s'. File may be empty or image-only.",
                task_id,
                original_filename,
            )
            return

        logger.info(
            "[Task %s] Loaded %d page(s) from '%s'.",
            task_id,
            len(raw_documents),
            original_filename,
        )

        # Step 3: Split document into overlapping semantic chunks
        splitter = _get_text_splitter()
        chunks: List[Document] = splitter.split_documents(raw_documents)

        if not chunks:
            logger.warning("[Task %s] Text splitter produced 0 chunks.", task_id)
            return

        logger.info(
            "[Task %s] Split %d page(s) into %d chunks.",
            task_id,
            len(raw_documents),
            len(chunks),
        )

        # Step 4: Enrich metadata on each chunk for traceability in RAG citations
        timestamp_str = datetime.now(timezone.utc).isoformat()
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "source": original_filename,
                    "task_id": task_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "ingested_at": timestamp_str,
                }
            )

        # Step 5: Embed chunks and persist to ChromaDB
        logger.info(
            "[Task %s] Generating embeddings and persisting %d chunks to ChromaDB...",
            task_id,
            len(chunks),
        )
        doc_ids = add_documents_to_chroma(chunks)

        elapsed_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            "[Task %s] Ingestion successfully completed in %.2f seconds! Indexed %d chunks for '%s'.",
            task_id,
            elapsed_duration,
            len(doc_ids),
            original_filename,
        )

    except Exception as exc:
        logger.exception(
            "[Task %s] Unhandled error during PDF ingestion for '%s': %s",
            task_id,
            original_filename,
            exc,
        )
    finally:
        # Step 6: Clean up the temporary upload file to prevent storage bloat
        if delete_temp_file and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug("[Task %s] Cleaned up temporary file: %s", task_id, file_path)
            except OSError as cleanup_err:
                logger.warning(
                    "[Task %s] Failed to remove temporary file '%s': %s",
                    task_id,
                    file_path,
                    cleanup_err,
                )
