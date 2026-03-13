"""
api/v1/endpoints/history.py - Conversation History Endpoint

GET /api/v1/history/{thread_id}
- Returns all messages for a given thread
"""

from fastapi import APIRouter, HTTPException
from schemas.chat import HistoryResponse, MessageResponse
from services.memory_service import get_all_messages

router = APIRouter()


@router.get("/history/{thread_id}", response_model=HistoryResponse)
def get_history(thread_id: str):
    """
    Retrieve all messages for a conversation thread.

    Returns messages in chronological order (oldest first).
    Each message has: role (user/assistant), content, and timestamp.
    """
    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id cannot be empty")

    messages = get_all_messages(thread_id)

    return HistoryResponse(
        thread_id=thread_id,
        messages=[
            MessageResponse(
                role=msg["role"],
                content=msg["content"],
                created_at=msg["created_at"]
            )
            for msg in messages
        ]
    )
