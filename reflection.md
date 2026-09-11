# Reflection: Building a Baseline Naive RAG Application

**Author:** Hiem Sreynit
**Course:** HRD Advanced Course - Artificial Intelligence (Week 5 Homework)
**Topic:** Naive RAG Fundamentals - Chat with Documents

### What Worked Well

Building the RAG application step by step helped me understand how the different components work together. Loading the three documents, splitting them into chunks, generating embeddings, and storing them in ChromaDB made the overall RAG workflow easier to understand. Using Ollama locally with `nomic-embed-text` for embeddings and `qwen2.5:0.5b-instruct` for generation also helped me see how a local model can be connected to a vector database. During testing, questions related to the documents could be answered using the retrieved context. The grounding prompt also helped the model stay focused on the provided documents instead of relying on outside information.

### What Was Harder Than Expected

One of the harder parts was getting the different components to work together correctly. I had to make sure the configuration, Ollama client, embedding model, ChromaDB, and retrieval code were all using the correct settings. I also learned that chunk size and overlap can affect retrieval quality. My application uses an 800-character chunk size with a 120-character overlap, so the chunks contain enough information while still being small enough for retrieval.

### Ideas for Improvement Using Advanced RAG Techniques

One improvement I would like to add in the future is **re-ranking**. Currently, the application retrieves the top 5 chunks from ChromaDB based on vector similarity and sends them to the generation model. However, the most similar chunks are not always the most useful ones. A re-ranker could examine the retrieved chunks again and place the most relevant information first before sending the final context to the LLM. This could improve the accuracy and relevance of the answers. Re-ranking is a common post-retrieval technique used in Advanced RAG systems.
