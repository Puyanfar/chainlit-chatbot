"""
Shared logic for writing FAQ pairs into the two Qdrant collections.
Used by the standalone FAQ API (api/main.py). Kept separate from app.py
so both the API and the chat app import the same client instances rather
than each creating their own.
"""

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
import uuid
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


async def ensure_collections() -> None:
    """Create both collections if they don't exist yet. Safe to call on every
    startup - does nothing if they're already there."""
    await config.initialize_embedding_size()
    if config.EMBEDDING_SIZE is None:
        raise RuntimeError(
            "Failed to determine embedding size for model '%s'" % config.EMBEDDING_MODEL
        )
    for name in (config.QDRANT_QUESTIONS_COLLECTION, config.QDRANT_QA_COLLECTION):
        exists = await qdrant_client.collection_exists(name)
        if not exists:
            await qdrant_client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=config.EMBEDDING_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in a single OpenAI API call."""
    response = await embedding_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def add_faq_pairs(pairs: list[dict]) -> list[str]:
    """
    pairs: list of {"question": str, "answer": str}.

    For each pair: generates a UUID4 id, embeds the question alone (for the
    questions collection) and the question+answer combined (for the qa
    collection), and upserts into both collections under the SAME point id -
    this is what lets retrieval later recognize a hit in either collection
    as belonging to the same underlying FAQ entry.

    Returns the generated ids, in the same order as the input pairs.
    """
    if not pairs:
        return []

    faq_ids = [str(uuid.uuid4()) for _ in pairs]
    questions = [p["question"] for p in pairs]
    combined_texts = [f"Q: {p['question']}\nA: {p['answer']}" for p in pairs]

    # Two embedding calls total, regardless of batch size - not one call per item.
    question_vectors = await embed_texts(questions)
    combined_vectors = await embed_texts(combined_texts)

    def make_payload(pair: dict, faq_id: str) -> dict:
        return {
            "question": pair["question"],
            "answer": pair["answer"],
            "faq_id": faq_id,
        }

    question_points = [
        models.PointStruct(id=faq_id, vector=vector, payload=make_payload(pair, faq_id))
        for faq_id, pair, vector in zip(faq_ids, pairs, question_vectors)
    ]
    qa_points = [
        models.PointStruct(id=faq_id, vector=vector, payload=make_payload(pair, faq_id))
        for faq_id, pair, vector in zip(faq_ids, pairs, combined_vectors)
    ]

    await qdrant_client.upsert(
        collection_name=config.QDRANT_QUESTIONS_COLLECTION, points=question_points
    )
    await qdrant_client.upsert(
        collection_name=config.QDRANT_QA_COLLECTION, points=qa_points
    )

    return faq_ids
