"""
generator.py
------------
Stage 6 of the Naive RAG Pipeline.

Uses retrieved document chunks as context
and asks the local Ollama LLM to generate an answer.
"""

from typing import List, Dict, Any, Optional

from config import settings
from ollama_client import default_client


LLM_MODEL = settings.LLM_MODEL
RELEVANCE_THRESHOLD = settings.RELEVANCE_THRESHOLD


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context for the LLM."""

    if not retrieved_chunks:
        return "No relevant context found."

    context = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        metadata = chunk.get("metadata", {})
        filename = metadata.get("filename", "document")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "").strip()

        context.append(
            f"[Source: {filename} | Chunk: {chunk_id}]\n"
            f"{text}"
        )

    return "\n\n".join(context)


def call_llm(
    prompt: str,
    system_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1
) -> str:
    """Send the prompt to the local Ollama model."""

    model = model or LLM_MODEL

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return default_client.chat_completion(
        messages,
        model=model,
        temperature=temperature
    )


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
    strict_check: bool = True
) -> str:
    """Generate an answer using only the retrieved documents."""

    # Check whether we have relevant information
    if strict_check:
        if not retrieved_chunks:
            return "I could not find this in your documents."

        max_similarity = max(
            (
                chunk.get("similarity", 0.0)
                for chunk in retrieved_chunks
            ),
            default=0.0
        )

        if max_similarity < RELEVANCE_THRESHOLD:
            return "I could not find this in your documents."

    # Prepare retrieved chunks
    context = format_context(retrieved_chunks)

    # Tell the LLM how it should answer
    system_prompt = (
        "You are a helpful assistant that answers questions "
        "using only the provided document context.\n\n"
        "Rules:\n"
        "1. Use only information from the provided documents.\n"
        "2. Do not use outside knowledge or make assumptions.\n"
        "3. If the answer is not in the documents, say: "
        "\"I could not find this in your documents.\"\n"
        "4. Keep the answer clear and concise."
    )

    # Build the final prompt
    user_prompt = (
        f"Context:\n"
        f"--------------------\n"
        f"{context}\n"
        f"--------------------\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    return call_llm(
        user_prompt,
        system_prompt,
        model=model
    )


if __name__ == "__main__":
    from retriever import retrieve_chunks

    print("=" * 50)
    print("Generator Test")
    print("=" * 50)

    # Test an in-document question
    question_1 = "How do I schedule a meeting on Cisco Webex?"

    print(f"\nQuestion: {question_1}")

    chunks = retrieve_chunks(question_1)
    answer = generate_answer(question_1, chunks)

    print(f"Answer:\n{answer}")

    # Test an out-of-document question
    question_2 = "What is the capital of France?"

    print(f"\nQuestion: {question_2}")

    chunks = retrieve_chunks(question_2)
    answer = generate_answer(question_2, chunks)

    print(f"Answer:\n{answer}")