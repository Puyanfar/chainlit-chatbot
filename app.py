from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import cast
import chainlit as cl
import logging
import config
from rag import retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = {"role": "system", "content": config.SYSTEM_PROMPT}
MAX_HISTORY_MESSAGES = config.MAX_HISTORY_MESSAGES
ERROR_MARKER = "⚠️"
FOLLOW_UP_ACTION_NAME = "follow_up_question"

client = AsyncOpenAI(
    base_url=config.API_ENDPOINT,
    api_key=config.API_KEY,
    timeout=config.REQUEST_TIMEOUT,
    max_retries=config.MAX_RETRIES,
)


def build_follow_up_actions(suggestions: list[dict]) -> list[cl.Action]:
    """One cl.Action per suggestion, all sharing the same action name (the
    callback below is registered against that name) - each carries its own
    question text in its payload so the callback knows which was clicked."""
    return [
        cl.Action(
            name=FOLLOW_UP_ACTION_NAME,
            payload={"question": suggestion["question"]},
            label=suggestion["question"],
        )
        for suggestion in suggestions
    ]


async def answer_question(question: str) -> None:
    """Core answering logic, shared by both a typed message and a clicked
    follow-up suggestion: retrieve, stream the model's response, update
    history, then offer any follow-up suggestions as clickable actions."""
    message_history = cl.user_session.get("message_history") or [SYSTEM_PROMPT]

    retrieval = await retrieve(question)
    retrieved_context = retrieval["context"]
    suggestions = retrieval["suggestions"]

    augmented_content = f"User Question:\n {question}\n\n"

    if retrieved_context:
        augmented_content += f"Reference Information:\n{retrieved_context}\n\n"
    else:
        augmented_content += "Reference Information:\n No relevant information was found in the knowledge base.\n\n"

    request_messages = message_history + [
        {"role": "user", "content": augmented_content}
    ]
    message_history.append({"role": "user", "content": question})

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

    if stream_succeeded and suggestions:
        actions = build_follow_up_actions(suggestions)
        await cl.Message(content="", actions=actions).send()


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [SYSTEM_PROMPT])


@cl.on_message
async def main(message: cl.Message):
    await answer_question(message.content)


@cl.action_callback(FOLLOW_UP_ACTION_NAME)
async def on_follow_up_click(action: cl.Action):
    question = action.payload.get("question", "")
    await action.remove()
    if question:
        await cl.Message(content=question, type="user_message").send()
        await answer_question(question)
