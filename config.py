import os
from dotenv import load_dotenv

load_dotenv()

# Endee Vector Database (local Docker)
ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8000/api/v1")

# Ollama (local LLM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Endee index name
INDEX_NAME = "interview_assistant"
