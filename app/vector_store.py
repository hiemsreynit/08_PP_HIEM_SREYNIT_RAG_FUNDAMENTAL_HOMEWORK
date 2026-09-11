"""
vector_store.py
---------------
Stage 4 of Naive RAG Pipeline: Vector Database Storage and Indexing.
Uses ChromaDB in persistent mode to store chunk texts, metadata, and embeddings,
and enables cosine/L2 semantic distance searching.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import settings


DEFAULT_DB_DIR = settings.CHROMA_DIR
DEFAULT_COLLECTION_NAME = settings.COLLECTION_NAME


def get_chroma_client(persist_directory: str = DEFAULT_DB_DIR) -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client.
    """
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_DB_DIR
) -> chromadb.Collection:
    """
    Gets an existing Chroma collection or creates a new one with cosine similarity.
    """
    if client is None:
        client = get_chroma_client(persist_directory)

    # Use cosine distance space
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_chunks_to_vector_store(
    collection: chromadb.Collection,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> int:
    """
    Adds text chunks and their corresponding embedding vectors to ChromaDB.
    Uses upsert to ensure idempotent indexing (no duplicates on re-runs).

    Args:
        collection: The ChromaDB collection object.
        chunks (List[Dict[str, Any]]): List of chunk objects from chunking.py.
        embeddings (List[List[float]]): List of vectors from embeddings.py.

    Returns:
        int: Number of chunks indexed.
    """
    if not chunks:
        print("[Warning] No chunks to index.")
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
        )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Chroma upsert updates existing or inserts new
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"[VectorStore] Successfully upserted {len(chunks)} chunks into collection '{collection.name}'.")
    return len(chunks)


def query_vector_store(
    collection: chromadb.Collection,
    query_vector: List[float],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Performs nearest-neighbor search in ChromaDB using the query vector.

    Args:
        collection: ChromaDB collection.
        query_vector (List[float]): Query embedding vector.
        top_k (int): Number of most similar chunks to retrieve.

    Returns:
        List[Dict[str, Any]]: Retrieved chunks sorted by relevance.
    """
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    if not results or not results["ids"] or not results["ids"][0]:
        return retrieved

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
        # With cosine distance: distance ranges 0.0 (identical) to 2.0 (opposite)
        # Cosine similarity is approx (1 - distance)
        similarity = round(1.0 - dist, 4)
        retrieved.append({
            "chunk_id": chunk_id,
            "text": doc,
            "metadata": meta,
            "distance": round(dist, 4),
            "similarity": similarity
        })

    return retrieved


def count_documents(collection: chromadb.Collection) -> int:
    """Returns total count of vectors stored in the collection."""
    return collection.count()


def reset_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_DB_DIR
) -> chromadb.Collection:
    """Deletes and recreates the collection."""
    if client is None:
        client = get_chroma_client(persist_directory)

    try:
        client.delete_collection(name=collection_name)
        print(f"[VectorStore] Existing collection '{collection_name}' deleted.")
    except Exception as e:
        # Collection might not exist yet on initial run
        pass

    return get_or_create_collection(client, collection_name, persist_directory)


if __name__ == "__main__":
    from ingestion import load_documents
    from chunking import chunk_all_documents
    from embeddings import get_embeddings_batch

    print("=" * 60)
    print("Stage 4: Vector Store Indexing Test")
    print("=" * 60)

    # 1. Load documents
    docs = load_documents(settings.DATA_DIR)
    print(f"Loaded {len(docs)} documents.")

    # 2. Chunk documents
    chunks = chunk_all_documents(
        docs,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    print(f"Produced {len(chunks)} chunks.")

    # 3. Generate embeddings
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = get_embeddings_batch(texts)

    # 4. Save to ChromaDB
    collection = get_or_create_collection()
    add_chunks_to_vector_store(collection, chunks, vectors)

    total_in_db = count_documents(collection)
    print(f"\nTotal chunks stored in persistent ChromaDB ('{DEFAULT_DB_DIR}'): {total_in_db}")
    print("Vector Store module successfully indexed all chunks!\n")
