"""
demo_vector_check.py
--------------------
Step 4 Verification Script:
Standalone check to test the vector store on its own before building the online RAG pipeline.
Embeds a test question, searches ChromaDB, and prints the top 3 results with similarity scores.
"""

from config import settings
from embeddings import get_embedding
from app.vector_store import get_or_create_collection, query_vector_store, count_documents


def run_demo_check(test_question: str = "How do I reset my forgotten PIN?", top_k: int = settings.DEFAULT_TOP_K):
    print("=" * 70)
    print("DEMO VECTOR CHECK: Testing ChromaDB Semantic Retrieval")
    print("=" * 70)

    # 1. Connect to ChromaDB
    collection = get_or_create_collection()
    total_chunks = count_documents(collection)
    print(f"Connected to ChromaDB Collection: '{collection.name}'")
    print(f"Total Chunks in Vector Store  : {total_chunks}")

    if total_chunks == 0:
        print("[Warning] Collection is empty! Please run 'python vector_store.py' first.")
        return

    # 2. Embed the test question
    print(f"\n[1] Embedding test question: '{test_question}'")
    query_vector = get_embedding(test_question)
    print(f"    Vector generated successfully (dimension: {len(query_vector)}).")

    # 3. Search vector database
    print(f"\n[2] Searching for top {top_k} most relevant chunks in ChromaDB...")
    results = query_vector_store(collection, query_vector, top_k=top_k)

    # 4. Display results
    print(f"\n[3] Top {len(results)} Matching Results:")
    print("-" * 70)

    for rank, item in enumerate(results, 1):
        meta = item["metadata"]
        print(f"Rank #{rank}:")
        print(f"  - Chunk ID   : {item['chunk_id']}")
        print(f"  - Source Doc : {meta.get('filename', 'unknown')}")
        print(f"  - Similarity : {item['similarity']} (Cosine distance: {item['distance']})")
        print(f"  - Text Content:")
        lines = item["text"].split("\n")
        indented_text = "\n".join(f"      {line}" for line in lines)
        print(f"{indented_text}\n")

    print("=" * 70)
    print("Check completed! If the top chunk matches the topic of your question,")
    print("your offline pipeline (Ingest -> Chunk -> Embed -> Store) is fully verified.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo_check()
