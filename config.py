import os
from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. Check your .env file."
        )
    return value


API_KEY: str = _require_env("API_KEY")
API_ENDPOINT: str = _require_env("API_ENDPOINT")
MODEL: str = _require_env("MODEL")

REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "faq_data")

RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.45"))

FAQ_DATA_PATH: str = os.getenv("FAQ_DATA_PATH", "data/faq_data.json")
