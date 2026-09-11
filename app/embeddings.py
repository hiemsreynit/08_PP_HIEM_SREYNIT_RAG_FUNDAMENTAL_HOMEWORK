"""
embeddings.py
-------------
Stage 3: Generate embeddings using Ollama.
"""

from typing import List, Optional

from app import config
from app.ollama_client import default_client


EMBED_MODEL = config.EMBED_MODEL


def get_embedding(
    text: str,
    model: Optional[str] = None
) -> List[float]:
    """Generate an embedding for one text."""

    return default_client.get_embedding(
        text,
        model=model or EMBED_MODEL
    )


def get_embeddings(
    texts: List[str],
    model: Optional[str] = None
) -> List[List[float]]:
    """Generate embeddings for multiple texts."""

    model = model or EMBED_MODEL

    return [
        get_embedding(text, model)
        for text in texts
    ]


if __name__ == "__main__":
    print("=" * 50)
    print("Embedding Test")
    print("=" * 50)

    text = "How do I schedule a conference call?"
    embedding = get_embedding(text)

    print(f"Model     : {EMBED_MODEL}")
    print(f"Text      : {text}")
    print(f"Vector size: {len(embedding)}")
    print(
        f"First 5 values: "
        f"{[round(x, 4) for x in embedding[:5]]}"
    )

    print("\nEmbedding test completed!")