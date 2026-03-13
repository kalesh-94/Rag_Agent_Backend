"""
schemas/chat.py - Request and Response Schemas

These are Pydantic models used by FastAPI to:
1. Validate incoming request bodies
2. Define the shape of responses
3. Auto-generate API documentation

FastAPI reads these and creates the /docs Swagger UI automatically.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────
# Request Schemas (what the client sends)
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request body for POST /ingest"""
    text: str = Field(
        ...,
        description="Raw document text to ingest into the RAG system",
        example="FastAPI is a modern web framework for building APIs with Python."
    )
    source: Optional[str] = Field(
        default="manual",
        description="Optional label for the source of this document",
        example="fastapi_docs"
    )


class ChatRequest(BaseModel):
    """Request body for POST /chat"""
    thread_id: str = Field(
        ...,
        description="Unique identifier for the conversation thread",
        example="user-123-session-abc"
    )
    message: str = Field(
        ...,
        description="The user's message/question",
        example="What is FastAPI?"
    )


# ─────────────────────────────────────────────
# Response Schemas (what we send back)
# ─────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response for POST /ingest"""
    message: str
    chunks_added: int


class ChatResponse(BaseModel):
    """Response for POST /chat"""
    thread_id: str
    answer: str


class MessageResponse(BaseModel):
    """Single message in history"""
    role: str
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    """Response for GET /history/{thread_id}"""
    thread_id: str
    messages: list[MessageResponse]
