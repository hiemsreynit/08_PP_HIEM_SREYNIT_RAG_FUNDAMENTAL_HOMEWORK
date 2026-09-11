"""
main.py
-------
Stage 8 of the Naive RAG Pipeline.

Provides an interactive terminal chat interface
for asking questions about the documents.
"""

from config import settings
from pipeline import RAGPipeline, setup_rag_system


def print_banner():
    """Display application information and available commands."""

    print(
        f"""
==================================================
        RAG - CHAT WITH YOUR DOCUMENTS
==================================================
LLM Model       : {settings.LLM_MODEL}
Embedding Model: {settings.EMBEDDING_MODEL}
Vector Database: ChromaDB
ChromaDB Path   : {settings.CHROMA_DIR}
Data Directory  : {settings.DATA_DIR}
--------------------------------------------------
Commands:
  - Ask a question and press Enter
  - 'exit' or 'quit'  → Exit the application
  - '/reindex'        → Re-index all documents
  - '/help'           → Show this help message
  - '/toggle-sources' → Show/hide retrieved sources
==================================================
"""
    )


def show_retrieved_chunks(chunks):
    """Display the chunks retrieved for the current question."""

    if not chunks:
        return

    print("\n" + "-" * 70)
    print("Retrieved Context")
    print("-" * 70)

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})

        filename = metadata.get("filename", "unknown")
        chunk_id = chunk.get("chunk_id", "N/A")
        similarity = chunk.get("similarity", 0.0)

        text = chunk.get("text", "").replace("\n", " ")
        preview = text[:140]

        print(
            f"[{i}] {filename} | "
            f"{chunk_id} | "
            f"Similarity: {similarity:.4f}"
        )
        print(f"    {preview}...")
        print()

    print("-" * 70)


def main():
    """Run the interactive RAG application."""

    print_banner()

    print("[Init] Starting RAG system...")

    pipeline = RAGPipeline(
        data_dir=settings.DATA_DIR
    )

    print("[Init] RAG system is ready!\n")

    show_sources = True

    while True:
        try:
            user_input = input("\nUser > ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()

        # Exit commands
        if command in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        # Show help
        if command in ("/help", "help"):
            print_banner()
            continue

        # Re-index documents
        if command == "/reindex":
            print(
                f"\n[Action] Re-indexing documents "
                f"from '{settings.DATA_DIR}'..."
            )

            setup_rag_system(
                data_dir=settings.DATA_DIR,
                force_reindex=True
            )

            print("[Action] Re-indexing completed!")
            continue

        # Show/hide retrieved sources
        if command == "/toggle-sources":
            show_sources = not show_sources
            status = "ON" if show_sources else "OFF"

            print(f"[Setting] Source display: {status}")
            continue

        # Run the RAG pipeline
        print("\n[Thinking] Searching documents...")

        result = pipeline.query(
            user_input,
            top_k=settings.DEFAULT_TOP_K
        )

        chunks = result["retrieved_chunks"]
        answer = result["answer"]

        # Display retrieved chunks
        if show_sources:
            show_retrieved_chunks(chunks)

        # Display final answer
        print(f"\nAssistant:\n{answer}")


if __name__ == "__main__":
    main()