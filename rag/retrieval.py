"""
Read side of the RAG system: given a user query, search both FAQ collections
and produce the context to ground an answer in, plus related follow-up
question suggestions - all from a single retrieval pass.
"""

import asyncio
import logging
import config
from rag.clients import embed_texts, qdrant_client
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def retrieve(query: str, exclude_ids: set[str] | None = None) -> dict:
    """Search both collections once and produce three things from the same
    result set:
      - "context": the FAQ text to ground the answer in (only entries that
        clear the strict RAG_SCORE_THRESHOLD).
      - "context_ids": the faq_ids behind that context - the caller uses this
        to track "entries already used", so they're not suggested again later.
      - "suggestions": other related FAQ questions worth offering as
        clickable follow-ups (entries that clear the looser
        RAG_SUGGESTION_SCORE_THRESHOLD, weren't already used for context this
        turn, AND aren't in exclude_ids - i.e. weren't already asked/used in
        an earlier turn this session).

    exclude_ids only affects the *suggestions* list, never grounding - if the
    user re-asks something close to a question they already covered, it can
    still ground a fresh answer; it just won't be offered again as a "you
    might also ask" suggestion.

    These two outputs are otherwise independent - the score gap between the
    two thresholds means there's a real middle ground: candidates related
    enough to suggest but not confident enough to answer with. In that case
    "context" comes back empty while "suggestions" still has entries - closer
    to a "did you mean one of these?" than plain silence. Suggestions are
    never separately generated - they're always a subset of what we already
    retrieved, so every suggestion is guaranteed to be a real, answerable
    FAQ entry.

    Returns {"context": str, "context_ids": [str, ...],
             "suggestions": [{"id": str, "question": str}, ...]}.
    context/context_ids/suggestions are all empty only when nothing clears
    even the loose threshold, or on a retrieval failure.
    """

    exclude_ids = exclude_ids or set()
    empty = {"context": "", "context_ids": [], "suggestions": []}

    try:
        [query_vector] = await embed_texts([query])

        fetch_limit = config.RAG_TOP_K + config.RAG_SUGGESTION_COUNT + len(exclude_ids)

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
    context_ids = list(used_ids)

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

    # Whatever's left in the ranked pool - already above the looser threshold,
    # already excluding what was used for the answer this turn, AND excluding
    # anything already asked/used in an earlier turn - becomes suggestions.
    suggestions = [
        {"id": faq_id, "question": payload["question"]}
        for faq_id, (score, payload) in ranked
        if faq_id not in used_ids and faq_id not in exclude_ids
    ][: config.RAG_SUGGESTION_COUNT]

    if suggestions:
        logger.info(
            "Suggesting %d follow-up question(s): %s",
            len(suggestions),
            [s["question"] for s in suggestions],
        )

    return {"context": context, "context_ids": context_ids, "suggestions": suggestions}


@dataclass
class RetrievalResult:
    augmented_content: str
    suggestions: list
    context_ids: list[str]


async def build_augmented_question(
    question: str, used_faq_ids: set[str]
) -> RetrievalResult:
    """Fetch RAG context for `question` and format it into the content
    block that gets sent to the model (never stored verbatim in history)."""
    retrieval = await retrieve(question, exclude_ids=used_faq_ids)
    retrieved_context = retrieval["context"]

    augmented_content = f"User Question:\n {question}\n\n"
    if retrieved_context:
        augmented_content += f"Reference Information:\n{retrieved_context}\n\n"
    else:
        augmented_content += (
            "Reference Information:\n"
            " No relevant information was found in the knowledge base.\n\n"
        )

    return RetrievalResult(
        augmented_content=augmented_content,
        suggestions=retrieval["suggestions"],
        context_ids=retrieval["context_ids"],
    )
