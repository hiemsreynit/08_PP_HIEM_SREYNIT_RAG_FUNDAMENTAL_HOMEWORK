"""
chunking.py
-----------
Stage 2: Split documents into smaller chunks.
"""

from typing import List, Dict, Any
from app import config
from app.ingestion import load_documents


def split_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks."""

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def chunk_document(
    document: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Split one document into structured chunks."""

    chunks = split_text(document["content"])

    return [
        {
            "chunk_id": f"{document['filename']}#c{i:02d}",
            "text": text,
            "metadata": {
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "chunk_index": i,
                "char_count": len(text)
            }
        }
        for i, text in enumerate(chunks)
    ]


def chunk_all_documents(
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Split all documents into chunks."""

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks


if __name__ == "__main__":
    print("=" * 50)
    print("Chunking Test")
    print("=" * 50)

    documents = load_documents()
    chunks = chunk_all_documents(documents)

    print(f"Documents : {len(documents)}")
    print(f"Chunk size: {config.CHUNK_SIZE}")
    print(f"Overlap   : {config.CHUNK_OVERLAP}")
    print(f"Total     : {len(chunks)} chunks")

    print("\nSample chunks:\n")

    for chunk in chunks[:3]:
        print(
            f"[{chunk['chunk_id']}] "
            f"{chunk['metadata']['char_count']} chars"
        )
        print(f"{chunk['text'][:150]}...\n")