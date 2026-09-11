"""
ingestion.py
------------
Stage 1 of the Naive RAG Pipeline.

Loads text documents from the data directory
and prepares them for the next stage.
"""

import os
from typing import List, Dict, Any

from app import config


def load_documents(
    data_dir: str = config.DATA_DIR
) -> List[Dict[str, Any]]:
    """Load all supported documents from the data directory."""

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found."
        )

    documents = []
    supported_files = (".txt", ".md")

    for filename in sorted(os.listdir(data_dir)):

        if not filename.lower().endswith(supported_files):
            continue

        file_path = os.path.join(data_dir, filename)

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:
                content = file.read().strip()

            if not content:
                print(
                    f"[Warning] Skipping empty file: {filename}"
                )
                continue

            documents.append({
                "doc_id": filename,
                "filename": filename,
                "content": content,
                "metadata": {
                    "source": file_path,
                    "filename": filename,
                    "char_count": len(content),
                    "word_count": len(content.split())
                }
            })

        except Exception as error:
            print(
                f"[Error] Failed to read "
                f"{file_path}: {error}"
            )

    return documents


if __name__ == "__main__":

    print("=" * 50)
    print("Document Ingestion Test")
    print("=" * 50)

    documents = load_documents()

    print(
        f"\nLoaded {len(documents)} document(s) "
        f"from '{config.DATA_DIR}/':\n"
    )

    for i, document in enumerate(documents, 1):

        metadata = document["metadata"]

        print(f"{i}. {document['filename']}")
        print(
            f"   Characters: "
            f"{metadata['char_count']}"
        )
        print(
            f"   Words     : "
            f"{metadata['word_count']}"
        )

        preview = (
            document["content"][:120]
            .replace("\n", " ")
        )

        print(
            f"   Preview   : "
            f"{preview}...\n"
        )