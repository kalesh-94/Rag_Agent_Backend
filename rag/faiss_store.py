"""
rag/faiss_store.py

FAISS Vector Store for RAG system.

This module:
1. Chunks documents
2. Creates embeddings
3. Stores vectors in FAISS
4. Retrieves relevant chunks for queries
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.config import settings

# -----------------------------
# Global in-memory state
# -----------------------------

faiss_index = None
stored_chunks: list[str] = []

# Load embedding model once
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)


# -----------------------------
# Embedding helper
# -----------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert text list → embedding vectors.
    """
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")


# -----------------------------
# Create FAISS index
# -----------------------------

def create_faiss_index(dimension: int) -> faiss.IndexFlatL2:
    """
    Create FAISS index using L2 distance.
    """
    print(f"Creating FAISS index with dimension {dimension}")
    return faiss.IndexFlatL2(dimension)


# -----------------------------
# Add documents
# -----------------------------

def add_documents(texts: list[str]):
    """
    Chunk → embed → add to FAISS
    """

    global faiss_index, stored_chunks

    if not texts:
        print(" No documents provided")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = splitter.split_text("\n\n".join(texts))

    if not chunks:
        print(" No chunks created")
        return

    print(f"Created {len(chunks)} chunks")

    # Embed
    vectors = embed_texts(chunks)

    # Initialize FAISS if needed
    if faiss_index is None:
        dimension = vectors.shape[1]
        faiss_index = create_faiss_index(dimension)

    # Add vectors
    faiss_index.add(vectors)

    # Save chunks
    stored_chunks.extend(chunks)

    print(f" FAISS now contains {faiss_index.ntotal} vectors")

    save_faiss_index()

    


# -----------------------------
# Retrieval
# -----------------------------

def retrieve_chunks(query: str, top_k: int = None) -> list[str]:
    """
    Retrieve most relevant chunks for query.
    """

    global faiss_index, stored_chunks

    if faiss_index is None or faiss_index.ntotal == 0:
        print(" FAISS index empty")
        return []

    top_k = top_k or settings.TOP_K_RESULTS

    # Prevent requesting more results than exist
    top_k = min(top_k, faiss_index.ntotal)

    query_vector = embed_texts([query])

    distances, indices = faiss_index.search(query_vector, top_k)

    results = []

    print("\n Retrieval Debug")
    print("---------------------")

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        chunk = stored_chunks[idx]
        distance = distances[0][i]

        print(f"\nResult {i+1} | Distance: {distance:.4f}")
        print(chunk[:200])

        results.append(chunk)

    print(f"\nRetrieved {len(results)} chunks")

    return results


# -----------------------------
# Save FAISS index
# -----------------------------

def save_faiss_index():

    global faiss_index, stored_chunks

    os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)

    faiss.write_index(
        faiss_index,
        f"{settings.FAISS_INDEX_PATH}/index.faiss"
    )

    with open(
        f"{settings.FAISS_INDEX_PATH}/chunks.pkl",
        "wb"
    ) as f:
        pickle.dump(stored_chunks, f)

    print("FAISS index saved")


# -----------------------------
# Load FAISS index
# -----------------------------

def load_faiss_index():

    global faiss_index, stored_chunks

    index_path = f"{settings.FAISS_INDEX_PATH}/index.faiss"
    chunks_path = f"{settings.FAISS_INDEX_PATH}/chunks.pkl"

    if os.path.exists(index_path) and os.path.exists(chunks_path):

        faiss_index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            stored_chunks = pickle.load(f)

        print(f" Loaded FAISS index with {faiss_index.ntotal} vectors")

    else:
        print("No FAISS index found — starting fresh")

        faiss_index = None
        stored_chunks = []