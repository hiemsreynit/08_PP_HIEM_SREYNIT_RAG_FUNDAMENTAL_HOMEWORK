"""
retriever.py
------------
Stage 5 of the Naive RAG Pipeline.

Converts a user's question into an embedding
and retrieves the most relevant chunks from ChromaDB.
"""

from typing import List, Dict, Any, Optional

from config import settings
from embeddings import get_embedding
from app.vector_store import get_or_create_collection, query_vector_store


class Retriever:
    """Retrieve relevant document chunks from the vector database."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        collection_name = collection_name or settings.COLLECTION_NAME
        persist_directory = persist_directory or settings.CHROMA_DIR

        self.collection = get_or_create_collection(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

    def retrieve(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find the most relevant chunks for a query."""

        if not query or not query.strip():
            return []

        # Convert the question into a vector
        query_embedding = get_embedding(query.strip())

        # Search the vector database
        results = query_vector_store(
            self.collection,
            query_embedding,
            top_k=top_k
        )

        # Remove results below the minimum similarity
        if min_similarity is not None:
            results = [
                result
                for result in results
                if result["similarity"] >= min_similarity
            ]

        return results


# Create the retriever only when it is needed
_default_retriever: Optional[Retriever] = None


def get_default_retriever() -> Retriever:
    """Return the default retriever instance."""
    global _default_retriever

    if _default_retriever is None:
        _default_retriever = Retriever()

    return _default_retriever


def retrieve_chunks(
    query: str,
    top_k: int = settings.DEFAULT_TOP_K,
    min_similarity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Retrieve chunks using the default retriever."""

    retriever = get_default_retriever()

    return retriever.retrieve(
        query,
        top_k=top_k,
        min_similarity=min_similarity
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Retriever Test")
    print("=" * 50)

    question = "How do I schedule a conference call on Cisco Webex?"

    print(f"\nQuestion: {question}\n")

    chunks = retrieve_chunks(question)

    if not chunks:
        print("No relevant chunks found.")
    else:
        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}] {chunk['chunk_id']}")
            print(f"    Similarity: {chunk['similarity']:.4f}")
            print(f"    Source: {chunk['metadata']['filename']}")
            print(f"    Preview: {chunk['text'][:120]}...")
            print()