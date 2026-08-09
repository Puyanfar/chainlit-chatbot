"""
Read side of the RAG system: given a user query, search both FAQ collections
and produce the context to ground an answer in, plus related follow-up
question suggestions - all from a single retrieval pass.
"""

import asyncio
import logging
import config
from rag.clients import embed_texts, qdrant_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def retrieve(query: str) -> dict:
    """Search both collections once and produce two things from the same
    result set:
      - "context": the FAQ text to ground the answer in (only entries that
        clear the strict RAG_SCORE_THRESHOLD).
      - "suggestions": other related FAQ questions worth offering as
        clickable follow-ups (entries that clear the looser
        RAG_SUGGESTION_SCORE_THRESHOLD but weren't already used for context).

    Returns {"context": str, "suggestions": [{"id": str, "question": str}, ...]}.
    Both are empty on failure or when nothing relevant is found.
    """
    empty = {"context": "", "suggestions": []}

    try:
        [query_vector] = await embed_texts([query])

        fetch_limit = config.RAG_TOP_K + config.RAG_SUGGESTION_COUNT

        questions_results, qa_results = await asyncio.gather(
            qdrant_client.query_points(
                collection_name=config.QDRANT_QUESTIONS_COLLECTION,
                query=query_vector,
                limit=fetch_limit,
                score_threshold=config.RAG_SUGGESTION_SCORE_THRESHOLD,
            ),
            qdrant_client.query_points(
                collection_name=config.QDRANT_QA_COLLECTION,
                query=query_vector,
                limit=fetch_limit,
                score_threshold=config.RAG_SUGGESTION_SCORE_THRESHOLD,
            ),
        )

    except Exception:
        logger.exception("Retrieval from Qdrant failed; continuing without RAG context")
        return empty

    merged: dict[str, tuple[float, dict]] = {}
    for results in (questions_results, qa_results):
        for point in results.points:
            if not point.payload:
                continue
            faq_id = str(point.id)
            if faq_id not in merged or point.score > merged[faq_id][0]:
                merged[faq_id] = (point.score, point.payload)

    if not merged:
        logger.info(
            "No candidates passed RAG_SUGGESTION_SCORE_THRESHOLD=%.2f",
            config.RAG_SUGGESTION_SCORE_THRESHOLD,
        )
        return empty

    ranked = sorted(merged.items(), key=lambda kv: kv[1][0], reverse=True)

    top_entries = [
        (faq_id, score, payload)
        for faq_id, (score, payload) in ranked
        if score >= config.RAG_SCORE_THRESHOLD
    ][: config.RAG_TOP_K]

    used_ids = {faq_id for faq_id, _, _ in top_entries}

    if top_entries:
        logging.info(
            "Using %d grounded chunk(s), scores: %s",
            len(top_entries),
            [round(score, 3) for _, score, _ in top_entries],
        )
        context_chunks = [
            f"Q: {payload['question']}\nA: {payload['answer']}"
            for _, _, payload in top_entries
        ]
        context = "\n\n---\n\n".join(context_chunks)
    else:
        logging.info(
            "No candidates passed RAG_SCORE_THRESHOLD=%.2f; no grounded context, "
            "but related suggestions may still apply.",
            config.RAG_SCORE_THRESHOLD,
        )
        context = ""

    # Whatever's left in the ranked pool (already above the looser threshold,
    # already excluding what was used for the answer) becomes suggestions.
    suggestions = [
        {"id": faq_id, "question": payload["question"]}
        for faq_id, (score, payload) in ranked
        if faq_id not in used_ids
    ][: config.RAG_SUGGESTION_COUNT]

    if suggestions:
        logger.info(
            "Suggesting %d follow-up question(s): %s",
            len(suggestions),
            [s["question"] for s in suggestions],
        )

    return {"context": context, "suggestions": suggestions}
