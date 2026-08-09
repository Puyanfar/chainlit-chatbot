from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import cast
from qdrant_client import AsyncQdrantClient
import chainlit as cl
import logging, asyncio
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = {"role": "system", "content": config.SYSTEM_PROMPT}
MAX_HISTORY_MESSAGES = config.MAX_HISTORY_MESSAGES
ERROR_MARKER = "⚠️"

client = AsyncOpenAI(
    base_url=config.API_ENDPOINT,
    api_key=config.API_KEY,
    timeout=config.REQUEST_TIMEOUT,
    max_retries=config.MAX_RETRIES,
)

qdrant_client = AsyncQdrantClient(url=config.QDRANT_URL)


async def retrieve_context(query: str) -> str:
    try:
        embedding_response = await client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=query,
        )
        query_vector = embedding_response.data[0].embedding

        questions_results, qa_results = await asyncio.gather(
            qdrant_client.query_points(
                collection_name=config.QDRANT_QUESTIONS_COLLECTION,
                query=query_vector,
                limit=config.RAG_TOP_K,
                score_threshold=config.RAG_SCORE_THRESHOLD,
            ),
            qdrant_client.query_points(
                collection_name=config.QDRANT_QA_COLLECTION,
                query=query_vector,
                limit=config.RAG_TOP_K,
                score_threshold=config.RAG_SCORE_THRESHOLD,
            ),
        )

    except Exception:
        logger.exception("Retrieval from Qdrant failed; continuing without RAG context")
        return ""

    merged: dict[str, tuple[float, dict]] = {}
    for results in (questions_results, qa_results):
        for point in results.points:
            if not point.payload:
                continue
            faq_id = str(point.id)
            if faq_id not in merged:
                merged[faq_id] = (point.score, point.payload)

    if not merged:
        logging.info(
            "No chunks passed RAG_SCORE_THRESHOLD=%.2f; skipping context.",
            config.RAG_SCORE_THRESHOLD,
        )
        return ""

    top_entries = sorted(merged.values(), key=lambda entry: entry[0], reverse=True)[
        : config.RAG_TOP_K
    ]

    logging.info(
        "Using %d merged chunk(s), scores: %s",
        len(top_entries),
        [round(score, 3) for score, _ in top_entries],
    )

    chunks = [
        f"Q: {payload['question']}\nA: {payload['answer']}"
        for _, payload in top_entries
    ]
    return "\n\n---\n\n".join(chunks)


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [SYSTEM_PROMPT])


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history") or [SYSTEM_PROMPT]

    retrieved_context = await retrieve_context(message.content)

    augmented_content = f"User Question:\n {message.content}\n\n"

    if retrieved_context:
        augmented_content += f"Reference Information:\n{retrieved_context}\n\n"
    else:
        augmented_content += (
            "Reference Information:\n No relevant information was found in the knowledge base.\n\n"
        )

    request_messages = message_history + [
        {"role": "user", "content": augmented_content}
    ]
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()
    full_response = ""

    stream_succeeded = False

    try:
        stream = await client.chat.completions.create(
            model=config.MODEL,
            messages=cast(list[ChatCompletionMessageParam], request_messages),
            stream=True,
        )

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    await msg.stream_token(delta)

        finally:
            await stream.close()

        stream_succeeded = True

    except Exception as e:
        logger.exception("Error while streaming response from model")

        if full_response:
            msg.content = (
                full_response
                + f"\n\n{ERROR_MARKER} ERROR! *Response cut off: something went wrong talking to the model: {e}"
            )

        else:
            msg.content = f"{ERROR_MARKER} ERROR! Sorry, something went wrong talking to the model: {e}"

    await msg.update()

    if stream_succeeded and full_response:
        message_history.append({"role": "assistant", "content": full_response})

        if len(message_history) > MAX_HISTORY_MESSAGES + 1:
            message_history = [SYSTEM_PROMPT] + message_history[-MAX_HISTORY_MESSAGES:]

        cl.user_session.set("message_history", message_history)
