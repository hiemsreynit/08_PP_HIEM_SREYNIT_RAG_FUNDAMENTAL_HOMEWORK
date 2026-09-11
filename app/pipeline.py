"""
pipeline.py
-----------
Stage 7 of the Naive RAG Pipeline.

Connects the different RAG stages together:

Offline:
Documents → Chunks → Embeddings → ChromaDB

Online:
Question → Retrieval → Generation → Answer
"""

from typing import Dict, Any, Optional

from config import settings

from ingestion import load_documents
from chunking import chunk_all_documents
from embeddings import get_embeddings

from app.vector_store import (
    get_or_create_collection,
    add_chunks_to_vector_store,
    count_documents,
    reset_collection
)

from retriever import get_default_retriever
from generator import generate_answer


def setup_rag_system(
    data_dir: str = settings.DATA_DIR,
    force_reindex: bool = False,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
) -> int:
    """
    Prepare the documents and store their embeddings in ChromaDB.
    """

    collection = get_or_create_collection()
    existing_count = count_documents(collection)

    # Use the existing database if it already contains documents
    if existing_count > 0 and not force_reindex:
        return existing_count

    # Clear the old data when re-indexing
    if force_reindex and existing_count > 0:
        print("[Pipeline] Resetting vector database...")
        collection = reset_collection()

    print("\n[Pipeline] Setting up knowledge base...")

    # 1. Load documents
    documents = load_documents(data_dir)

    if not documents:
        print(
            f"[Pipeline] No documents found in '{data_dir}'."
        )
        return 0

    print(
        f"[Pipeline] Loaded {len(documents)} document(s)."
    )

    # 2. Split documents into chunks
    chunks = chunk_all_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy="recursive"
    )

    print(
        f"[Pipeline] Created {len(chunks)} chunk(s)."
    )

    # 3. Generate embeddings
    texts = [chunk["text"] for chunk in chunks]

    print(
        f"[Pipeline] Generating embeddings "
        f"with {settings.EMBEDDING_MODEL}..."
    )

    vectors = get_embeddings(texts)

    # 4. Store chunks and vectors in ChromaDB
    add_chunks_to_vector_store(
        collection,
        chunks,
        vectors
    )

    total = count_documents(collection)

    print(
        f"[Pipeline] Knowledge base ready. "
        f"{total} chunk(s) indexed."
    )

    return total


class RAGPipeline:
    """Main interface for the complete RAG system."""

    def __init__(
        self,
        data_dir: Optional[str] = None
    ):
        self.data_dir = data_dir or settings.DATA_DIR

        # Make sure the vector database is ready
        setup_rag_system(
            data_dir=self.data_dir
        )

        # Create the retriever
        self.retriever = get_default_retriever()

    def query(
        self,
        question: str,
        top_k: int = settings.DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Run a question through the RAG pipeline.

        Question
            ↓
        Retrieve relevant chunks
            ↓
        Generate grounded answer
        """

        question = question.strip()

        if not question:
            return {
                "question": question,
                "answer": "Please provide a valid question.",
                "retrieved_chunks": []
            }

        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        # Generate the final answer
        answer = generate_answer(
            question,
            chunks
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks
        }


# Shared pipeline instance
_default_pipeline: Optional[RAGPipeline] = None


def get_pipeline(
    data_dir: Optional[str] = None
) -> RAGPipeline:
    """Return the shared RAG pipeline."""

    global _default_pipeline

    if _default_pipeline is None:
        _default_pipeline = RAGPipeline(
            data_dir=data_dir or settings.DATA_DIR
        )

    return _default_pipeline


def query_rag(
    question: str,
    top_k: int = settings.DEFAULT_TOP_K
) -> Dict[str, Any]:
    """Quick way to query the RAG system."""

    pipeline = get_pipeline()

    return pipeline.query(
        question,
        top_k=top_k
    )


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Pipeline Test")
    print("=" * 50)

    question = (
        "How do I schedule a conference call "
        "on Cisco Webex?"
    )

    print(f"\nQuestion: {question}")

    result = query_rag(question)

    print("\nRetrieved Chunks:")
    print(
        f"Found {len(result['retrieved_chunks'])} chunk(s)"
    )

    for i, chunk in enumerate(
        result["retrieved_chunks"],
        1
    ):
        print(
            f"[{i}] "
            f"{chunk['chunk_id']} | "
            f"Similarity: {chunk['similarity']:.4f}"
        )

    print("\nAnswer:")
    print(result["answer"])