"""
main.py - FastAPI Application Entry Point

This is where the app starts. It:
1. Creates the FastAPI app
2. Sets up the database tables on startup
3. Loads the FAISS index on startup
4. Registers all API routes
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager

from db.session import init_db
from rag.faiss_store import load_faiss_index
from  app.api.v1.endpoints import chat, ingest, history
from core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    - startup: create DB tables, load FAISS index
    - shutdown: nothing to do here
    """
    print("Starting up RAG backend...")

    # Create SQLite tables if they don't exist
    init_db()
    print("Database initialized")

    # Load FAISS index from disk (if it exists)
    load_faiss_index()
    print("FAISS index loaded")

    yield  # App runs here

    print("Shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    description="Thread-aware RAG Chat Backend using FAISS + OLLama + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])


@app.get("/")
def root():
    """ check endpoint"""
    return {"message": "RAG Backend is running!!!!!!!!", "status": "ok"}



