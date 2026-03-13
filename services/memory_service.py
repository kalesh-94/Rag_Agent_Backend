"""
services/memory_service.py - Conversation Memory using SQLite

This service handles all database operations for chat history.

WHY MEMORY:
- Without memory, each question is answered without context
- With memory, the AI knows what was said earlier in the conversation
- We store history per "thread_id" so multiple conversations are isolated
"""

from db.session import get_connection
from core.config import settings


def ensure_thread_exists(thread_id: str):
    """
    Insert a thread into the database if it doesn't already exist.
    Called before adding any message to a thread.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # INSERT OR IGNORE = only insert if the thread_id isn't already there
    cursor.execute(
        "INSERT OR IGNORE INTO threads (id) VALUES (?)",
        (thread_id,)
    )

    conn.commit()
    conn.close()


def save_message(thread_id: str, role: str, content: str):
    """
    Save a message to the database.

    Args:
        thread_id: which conversation this belongs to
        role: "user" or "assistant"
        content: the message text
    """
    ensure_thread_exists(thread_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content)
    )

    conn.commit()
    conn.close()


def get_recent_history(thread_id: str, limit: int = None) -> list[dict]:
    """
    Fetch the last N messages for a thread.

    WHY LIMIT:
    - LLMs have token limits
    - Sending too much history = slower responses and possible errors
    - Last 5 messages is usually enough for context

    Returns: list of dicts like [{"role": "user", "content": "..."}]
    """
    limit = limit or settings.MAX_HISTORY_MESSAGES

    conn = get_connection()
    cursor = conn.cursor()

    # Get last N messages ordered by time
    # We use a subquery to get the latest messages, then re-order them oldest-first
    cursor.execute("""
        SELECT role, content FROM (
            SELECT role, content, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        )
        ORDER BY created_at ASC
    """, (thread_id, limit))

    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to plain dicts
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_all_messages(thread_id: str) -> list[dict]:
    """
    Fetch ALL messages for a thread (used for /history endpoint).

    Returns: list of dicts with role, content, and created_at
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, content, created_at
        FROM messages
        WHERE thread_id = ?
        ORDER BY created_at ASC
    """, (thread_id,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]
