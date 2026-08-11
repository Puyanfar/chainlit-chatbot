from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
)
from typing import cast
import chainlit as cl
import logging, json
import config
from rag import retrieve
from mcp import ClientSession
from mcp.types import Tool, ListToolsResult, ToolUseContent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: ChatCompletionMessageParam = {
    "role": "system",
    "content": config.SYSTEM_PROMPT,
}
MAX_HISTORY_MESSAGES: int = config.MAX_HISTORY_MESSAGES
ERROR_MARKER: str = "⚠️"
FOLLOW_UP_ACTION_NAME: str = "follow_up_question"
ASSISTANT_AUTHOR: str = "assistant"

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


async def clear_active_follow_ups() -> None:
    """Remove the buttons from whatever follow-up suggestion message is
    currently outstanding, if any. Called at the very start of handling any
    new question - typed or clicked - so there's only ever one live set of
    suggestions on screen at a time, regardless of how the previous round
    was resolved."""

    active_message = cl.user_session.get("active_follow_up_message")
    if active_message:
        await active_message.remove()
        cl.user_session.set("active_follow_up_message", None)


async def answer_question(question: str) -> str:
    """Core answering logic, shared by both a typed message and a clicked
    follow-up suggestion: retrieve, stream the model's response, update
    history, then offer any follow-up suggestions as clickable actions."""

    await clear_active_follow_ups()

    message_history: list[ChatCompletionMessageParam] = cl.user_session.get(
        "message_history"
    ) or [SYSTEM_PROMPT]
    used_faq_ids: set[str] = cl.user_session.get("used_faq_ids") or set()
    mcp_tools: dict[str, list[dict]] = cast(
        dict[str, list[dict]], cl.user_session.get("mcp_tools", {})
    )

    tools = cast(
        list[ChatCompletionFunctionToolParam],
        [
            tool
            for connection_name, tool_list in mcp_tools.items()
            for tool in tool_list
        ],
    )
    # logger.info(f"Tools available for this session: {tools}")

    retrieval = await retrieve(question, exclude_ids=used_faq_ids)
    retrieved_context = retrieval["context"]
    suggestions = retrieval["suggestions"]
    context_ids = retrieval["context_ids"]

    augmented_content = f"User Question:\n {question}\n\n"

    if retrieved_context:
        augmented_content += f"Reference Information:\n{retrieved_context}\n\n"
    else:
        augmented_content += "Reference Information:\n No relevant information was found in the knowledge base.\n\n"

    request_messages = message_history + [
        {"role": "user", "content": augmented_content}
    ]
    message_history.append({"role": "user", "content": question})

    msg = cl.Message(content="", author=ASSISTANT_AUTHOR)
    await msg.send()
    full_response = ""

    stream_succeeded = False

    try:
        async with client.chat.completions.stream(
            model=config.MODEL,
            messages=request_messages,
            # tools=tools,
        ) as stream:
            async for event in stream:
                if event.type == "content.delta" and event.delta:
                    await msg.stream_token(event.delta)

            final_completion = await stream.get_final_completion()

        full_response = final_completion.choices[0].message.content or ""
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

    # Whatever FAQ entries grounded this answer are now "used" - never
    # suggest them again as a follow-up for the rest of this session.
    if context_ids:
        cl.user_session.set("used_faq_ids", used_faq_ids | set(context_ids))

    if stream_succeeded and suggestions:
        actions = build_follow_up_actions(suggestions)
        follow_up_message = cl.Message(
            content="", actions=actions, author=ASSISTANT_AUTHOR
        )
        await follow_up_message.send()
        cl.user_session.set("active_follow_up_message", follow_up_message)

    return full_response


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    # List available tools
    result: ListToolsResult = await session.list_tools()

    # Process tool metadata
    tools: list[dict] = cast(
        list[dict],
        [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ],
    )

    # Store tools for later use
    mcp_tools: dict[str, list[dict]] = cast(
        dict[str, list[dict]], cl.user_session.get("mcp_tools", {})
    )
    mcp_tools[connection.name] = tools
    # logger.info(f"Available tools for connection '{connection.name}': {tools}")
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.step(type="tool")
async def call_tool(tool_use: ToolUseContent) -> str:
    tool_name = tool_use.name
    tool_input = tool_use.input

    current_step = cl.context.current_step
    if current_step is not None:
        current_step.name = tool_name

    # Identify which mcp is used
    mcp_tools: dict[str, list[dict]] = cast(
        dict[str, list[dict]], cl.user_session.get("mcp_tools", {})
    )
    mcp_name = None

    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break

    error_output = json.dumps(
        {"error": f"Tool {tool_name} not found in any MCP connection"}
    )
    if not mcp_name:
        if current_step is not None:
            current_step.output = error_output
        return error_output

    mcp_sessions = getattr(cl.context.session, "mcp_sessions", {}) or {}
    mcp_session = mcp_sessions.get(mcp_name)

    if not mcp_session:
        error_output = json.dumps(
            {"error": f"MCP {mcp_name} not found in any MCP connection"}
        )
        if current_step is not None:
            current_step.output = error_output
        return error_output

    try:
        result = await mcp_session.call_tool(tool_name, tool_input)
        if current_step is not None:
            current_step.output = result
        return result
    except Exception as e:
        error_output = json.dumps({"error": str(e)})
        if current_step is not None:
            current_step.output = error_output
        return error_output


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [SYSTEM_PROMPT])
    cl.user_session.set("active_follow_up_message", None)
    cl.user_session.set("used_faq_ids", set())


@cl.on_message
async def main(message: cl.Message):
    await answer_question(message.content)


@cl.action_callback(FOLLOW_UP_ACTION_NAME)
async def on_follow_up_click(action: cl.Action):
    question = action.payload.get("question", "")
    if question:
        # A callback-triggered message doesn't go through the normal user-input
        # path, so nothing puts it in the UI on its own - explicitly send it as
        # a user-authored bubble before generating the answer, so the click
        # reads exactly like the user having typed the question themselves.
        # (The clicked button itself, and any other stale ones, get cleared
        # inside answer_question via clear_active_follow_ups().)
        await cl.Message(content=question, type="user_message").send()
        await answer_question(question)
