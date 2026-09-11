# --- Models ---
# Models must already be pulled in Ollama:
# ollama pull nomic-embed-text
# ollama pull qwen2.5:0.5b-instruct

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen2.5:0.5b-instruct"


# --- Storage ---
DATA_DIR = "data"                  # where source documents are stored
CHROMA_DB_DIR = "chroma_db"        # where the vector database is persisted
COLLECTION_NAME = "documents"


# --- Chunking ---
CHUNK_SIZE = 800                   # characters per chunk
CHUNK_OVERLAP = 120                # characters shared between chunks


# --- Retrieval ---
TOP_K = 5                           # number of chunks retrieved per question


# --- Generation ---
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the context, "
    'say "I don\'t have enough information in the documents to answer that." '
    "Do not use outside knowledge. Cite the source file name(s) you used."
)