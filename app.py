from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import cast
from qdrant_client import AsyncQdrantClient
import chainlit as cl
import logging
import config

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

        results = await qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=config.RAG_TOP_K,
            score_threshold=config.RAG_SCORE_THRESHOLD,
        )

    except Exception:
        logging.exception(
            "Retrieval from Qdrant failed; continuing without RAG context"
        )
        return ""

    if not results.points:
        return ""

    chunks = [point.payload["context"] for point in results.points if point.payload]
    return "\n\n---\n\n".join(chunks)


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [SYSTEM_PROMPT])


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history") or [SYSTEM_PROMPT]

    retrieved_context = await retrieve_context(message.content)

    if retrieved_context:
        print(retrieved_context)
        augmented_content = (
            "Use the following context to answer the question if relevant. "
            "If the context doesn't contain the answer, say so and answer from "
            "your own knowledge if you can.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {message.content}"
        )
    else:
        augmented_content = message.content
 
    request_messages = message_history + [{"role": "user", "content": augmented_content}]
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
        logging.exception("Error while streaming response from model")

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
