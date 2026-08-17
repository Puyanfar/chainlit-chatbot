"""
The one place the embedding OpenAI client and the Qdrant client are
constructed. Both retrieval.py and ingestion.py import from here instead of
each creating their own - so connection settings, auth, etc. only ever need
to change in one place.
"""

from langfuse.openai import AsyncOpenAI  # type: ignore
from langfuse import observe
from qdrant_client import AsyncQdrantClient
import config

embedding_client = AsyncOpenAI(
    base_url=config.API_ENDPOINT,
    api_key=config.API_KEY,
    timeout=config.REQUEST_TIMEOUT,
    max_retries=config.MAX_RETRIES,
)

qdrant_client = AsyncQdrantClient(
    url=config.QDRANT_URL,
)


@observe
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in a single OpenAI API call."""
    response = await embedding_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
