"""
core/config.py - Application Configuration

Central place for all application settings.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    # Application
    APP_NAME: str = "Thread-Aware RAG Backend"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TIMEOUT: int = 120

    # FAISS storage
    FAISS_INDEX_PATH: str = "faiss_index"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # SQLite
    SQLITE_DB_PATH: str = "chat_memory.db"

    # Chunking configuration
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 100

    # Retrieval
    TOP_K_RESULTS: int = 5

    # Conversation history
    MAX_HISTORY_MESSAGES: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()