"""
api/v1/endpoints/ingest.py

POST /api/v1/ingest

Supports:
1) Raw text ingestion via JSON
2) File upload (.txt, .pdf, .docx)

Text will be:
- chunked (500 chars, 50 overlap)
- embedded
- stored in FAISS
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.chat import IngestResponse
from rag.faiss_store import add_documents, stored_chunks

from docx import Document
from pypdf import PdfReader

router = APIRouter()


def extract_text(file: UploadFile) -> str:
    """
    Extract text from uploaded files.
    Supports txt, pdf, docx.
    """

    filename = file.filename.lower()

    # TXT
    if filename.endswith(".txt"):
        return file.file.read().decode("utf-8")

    # DOCX
    elif filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join([p.text for p in doc.paragraphs])

    # PDF
    elif filename.endswith(".pdf"):
        pdf = PdfReader(file.file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(None),
    text: str = Form(None),
    source: str = Form("manual"),
):
    """
    Ingest document into FAISS.

    Either:
    - upload a file
    OR
    - send raw text
    """

    if not file and not text:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or text"
        )

    try:
        # Extract text
        if file:
            extracted_text = extract_text(file)
        else:
            extracted_text = text

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text found"
            )

        # Count before ingestion
        chunks_before = len(stored_chunks)

        # Add document to FAISS
        add_documents([extracted_text])

        # Count added chunks
        chunks_added = len(stored_chunks) - chunks_before

        return IngestResponse(
            message=f"Document ingested successfully from source: {source}",
            chunks_added=chunks_added,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))