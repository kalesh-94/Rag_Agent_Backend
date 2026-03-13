"""
db/session.py - SQLite Database Setup

We use raw sqlite3 (no ORM like SQLAlchemy) to keep things simple.

Tables:
  - threads: stores each conversation thread
  - messages: stores each message (user/assistant) linked to a thread
"""

import sqlite3
from core.config import settings


def get_connection():
    """
    Returns a new SQLite connection.
    Call this whenever you need to run a query.
    Always close it after you're done (or use 'with' statement).
    """
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access: row["column_name"]
    return conn


def init_db():
    """
    Creates all required tables if they don't exist.
    Called once at startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create threads table
    # Each thread = one conversation session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,          -- UUID or any string ID
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create messages table
    # Each message belongs to a thread
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,      -- links to threads.id
            role TEXT NOT NULL,           -- "user" or "assistant"
            content TEXT NOT NULL,        -- the actual message text
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads(id)
        )
    """)

    conn.commit()
    conn.close()
    print(" DB tables ready: threads, messages")
