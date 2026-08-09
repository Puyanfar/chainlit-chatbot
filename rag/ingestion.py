"""
Write side of the RAG system: turn {"question", "answer"} pairs into vectors
in both Qdrant collections. Used by the standalone FAQ API (api/main.py).
"""

from qdrant_client import models
from rag.clients import embed_texts, qdrant_client
import uuid
import config

FAQ_ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


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


def faq_id_for(question: str) -> str:
    """Deterministic id derived from the question text. Same question (after
    normalizing case/whitespace) always yields the same id, so re-adding it
    is an update, not a duplicate. A changed answer for the same question
    keeps the same id too - only the payload/vectors change."""
    normalized = question.strip().lower()
    return str(uuid.uuid5(FAQ_ID_NAMESPACE, normalized))


async def _existing_ids(faq_ids: list[str]) -> set[str]:
    """Which of these ids are already present in the questions collection.
    Used only to report created-vs-updated back to the caller - the upsert
    itself works correctly either way."""
    if not faq_ids:
        return set()

    records = await qdrant_client.retrieve(
        collection_name=config.QDRANT_QUESTIONS_COLLECTION,
        ids=faq_ids,
        with_payload=False,
        with_vectors=False,
    )
    return {str(r.id) for r in records}


async def add_faq_pairs(pairs: list[dict]) -> list[dict]:
    """
    pairs: list of {"question": str, "answer": str}.

    For each pair: derives a deterministic id from the question, embeds the
    question alone (for the questions collection) and the question+answer
    combined (for the qa collection), and upserts into both collections
    under the SAME point id - this is what lets retrieval later recognize a
    hit in either collection as belonging to the same underlying FAQ entry.

    Returns [{"id": str, "question": str, "answer": str, "updated": bool}, ...]
    in the same order as the input pairs. "updated" is True if that id
    already existed before this call (i.e. this was an edit, not a new entry).
    """
    if not pairs:
        return []

    faq_ids = [faq_id_for(p["question"]) for p in pairs]
    questions = [p["question"] for p in pairs]
    combined_texts = [f"Q: {p['question']}\nA: {p['answer']}" for p in pairs]

    existing_ids = await _existing_ids(faq_ids)

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

    return [
        {
            "id": faq_id,
            "question": pair["question"],
            "answer": pair["answer"],
            "updated": faq_id in existing_ids,
        }
        for faq_id, pair in zip(faq_ids, pairs)
    ]
