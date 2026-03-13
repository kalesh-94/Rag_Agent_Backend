"""
api/v1/endpoints/chat.py - Chat Endpoint

POST /api/v1/chat
- Receives thread_id and message
- Runs LangGraph pipeline
- Returns AI-generated answer
"""

from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from rag.graph import run_chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message and get an AI-generated response.

    The pipeline:
    1. Load conversation history for this thread_id
    2. Find relevant document chunks from FAISS
    3. Build a structured prompt
    4. Generate answer via OLLama (mistral)
    5. Save messages to SQLite
    6. Return the answer

    thread_id can be any string — use it to separate conversations.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not request.thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id cannot be empty")

    # Run the full LangGraph pipeline
    answer = run_chat(
        thread_id=request.thread_id,
        user_input=request.message
    )

    return ChatResponse(
        thread_id=request.thread_id,
        answer=answer
    )
