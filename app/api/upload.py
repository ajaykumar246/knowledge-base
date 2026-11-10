"""
Document Upload and Ingestion Controller.

Exposes the POST /upload endpoint which accepts PDF files, saves them
temporarily, and enqueues asynchronous document parsing and embedding
via FastAPI BackgroundTasks.
"""

import logging
import os
import shutil
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.services.ingestion import process_pdf_background

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Ingestion"])

# Ensure temporary uploads directory exists
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class UploadResponse(BaseModel):
    """Schema for document upload response."""

    status: str
    message: str
    task_id: str
    filename: str
    file_size_bytes: int


try:
    import multipart  # Verify python-multipart is available for FastAPI Form/File handling
    MULTIPART_AVAILABLE = True
except ImportError:
    MULTIPART_AVAILABLE = False


if MULTIPART_AVAILABLE:
    @router.post(
        "/upload",
        response_model=UploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Upload PDF for Asynchronous Ingestion",
        description=(
            "Accepts a PDF document, stores it temporarily, and triggers an asynchronous "
            "background task to parse pages with PyPDFLoader, chunk text with "
            "RecursiveCharacterTextSplitter, generate Hugging Face embeddings, and index into ChromaDB."
        ),
    )
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="PDF document to be indexed into knowledge base"),
    ) -> UploadResponse:
        """
        Handles PDF file upload and schedules background parsing and vector indexing.
        """
        # 1. Validate file extension and MIME type
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must have a valid filename.",
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type for '{file.filename}'. Only .pdf files are supported.",
            )

        # 2. Generate unique tracking ID and temporary file path
        task_id = str(uuid.uuid4())
        safe_filename = os.path.basename(file.filename)
        temp_file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{safe_filename}")

        # 3. Stream uploaded file contents to disk in chunks to optimize memory usage
        file_size = 0
        try:
            with open(temp_file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):  # 1MB buffer
                    buffer.write(chunk)
                    file_size += len(chunk)

            logger.info(
                "Saved uploaded file '%s' (%d bytes) to '%s'",
                file.filename,
                file_size,
                temp_file_path,
            )
        except Exception as exc:
            logger.error("Failed to write uploaded file to disk: %s", exc)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store file on disk: {exc}",
            ) from exc
        finally:
            await file.close()

        # 4. Enqueue non-blocking background task
        background_tasks.add_task(
            process_pdf_background,
            file_path=temp_file_path,
            original_filename=safe_filename,
            task_id=task_id,
            delete_temp_file=True,
        )

        logger.info(
            "Dispatched background ingestion task '%s' for '%s'",
            task_id,
            safe_filename,
        )

        # 5. Return immediate acknowledgment to client
        return UploadResponse(
            status="accepted",
            message=(
                f"File '{safe_filename}' uploaded successfully. "
                "Parsing and embedding have started in the background."
            ),
            task_id=task_id,
            filename=safe_filename,
            file_size_bytes=file_size,
        )

else:
    @router.post(
        "/upload",
        response_model=UploadResponse,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        summary="Upload PDF (python-multipart dependency missing)",
        description="Requires python-multipart to accept multipart/form-data file uploads.",
    )
    async def upload_document_missing_dependency() -> UploadResponse:
        """Fallback endpoint triggered if python-multipart has not yet been installed."""
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The 'python-multipart' library is required for file uploads. Install via: pip install python-multipart",
        )
