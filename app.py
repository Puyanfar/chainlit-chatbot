from dataclasses import dataclass
from openai import AsyncOpenAI, APIError
from openai.types.responses import Response, ResponseInputParam, FunctionToolParam, ResponseFunctionToolCall
from mcp import ClientSession
from mcp.types import ListToolsResult, ToolUseContent
from typing import cast
from rag import build_augmented_question
from helpers import (
    get_message_history,
    save_message_history,
    get_used_faq_ids,
    mark_faq_ids_used,
    send_follow_up_suggestions,
    clear_active_follow_ups,
    mcp_tool_to_openai_tool,
)
import chainlit as cl
import logging, json, config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = config.SYSTEM_PROMPT
ERROR_MARKER: str = "⚠️"
FOLLOW_UP_ACTION_NAME: str = "follow_up_question"
ASSISTANT_AUTHOR: str = "assistant"

client = AsyncOpenAI(
    base_url=config.API_ENDPOINT,
    api_key=config.API_KEY,
    timeout=config.REQUEST_TIMEOUT,
    max_retries=config.MAX_RETRIES,
)


@dataclass
class StreamResult:
    text: str
    response: Response | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.response is not None


async def stream_assistant_reply(
    response_stream_message: cl.Message, input_items: ResponseInputParam
) -> StreamResult:
    """Stream a Responses API reply into an existing Chainlit message.

    Tokens are pushed to `response_stream_message` as they arrive via `stream_token`
    (Chainlit appends them to `response_stream_message.content` itself). Returns whatever
    text was produced even on failure, so the caller can show a
    "cut off" message instead of losing partial output.
    """
    full_text = ""
    final_response: Response | None = None
    working_inputs = list(input_items)
    mcp_tools = cl.user_session.get("mcp_tools", {}) or {}
    tools = [
        tool for connection_name, tools_list in mcp_tools.items() for tool in tools_list
    ]

    try:
        async with client.responses.stream(
            model=config.MODEL,
            instructions=SYSTEM_PROMPT,
            input=working_inputs,
            tools=tools,
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    full_text += event.delta
                    await response_stream_message.stream_token(event.delta)
                elif (
                    event.type == "response.output_item.done"
                    and event.item.type == "function_call"
                ):
                    item: ResponseFunctionToolCall = event.item
                    arguments = json.loads(item.arguments)
                    tool_name = item.name
                    call_id = item.call_id
                    logger.info(f"function call request by the model: item={item}, tool name={tool_name}, arguments={arguments}, call id={call_id}")
                    working_inputs.append({"type": "function_call", "call_id": call_id, "name": tool_name, "arguments": item.arguments})
                    tool_use = {"name":tool_name, "arguments": arguments, "call_id": call_id}
                    await call_tool(
                        tool_use,
                        working_inputs,
                        response_stream_message,
                    )
                elif event.type == "response.error":
                    # get_final_response()/the context manager will also
                    # surface this; logging here just keeps the timeline.
                    logger.error("Responses API stream error: %s", event.error)

            final_response = await stream.get_final_response()

        if final_response.status != "completed":
            detail = final_response.error.message if final_response.error else None
            raise RuntimeError(
                f"model response ended with status={final_response.status!r}"
                + (f": {detail}" if detail else "")
            )

        return StreamResult(text=full_text, response=final_response, error=None)

    except APIError as e:
        logger.exception(f"OpenAI API error while streaming response: {e}")
        return StreamResult(text=full_text, response=final_response, error=e)
    except Exception as e:
        logger.exception(f"Unexpected error while streaming response: {e}")
        return StreamResult(text=full_text, response=final_response, error=e)


async def answer_question(question: str) -> Response | None:
    """Core answering logic, shared by both a typed message and a clicked
    follow-up suggestion: retrieve, stream the model's response, update
    history, then offer any follow-up suggestions as clickable actions."""

    await clear_active_follow_ups()

    history = get_message_history()
    used_faq_ids = get_used_faq_ids()

    retrieval = await build_augmented_question(question, used_faq_ids)

    request_input: ResponseInputParam = [
        *history,
        {"role": "user", "content": retrieval.augmented_content},
    ]

    msg = cl.Message(content="", author=ASSISTANT_AUTHOR)
    await msg.send()

    result = await stream_assistant_reply(msg, request_input)

    if not result.ok:
        if result.text:
            msg.content = (
                result.text + f"\n\n{ERROR_MARKER} ERROR! *Response cut off: "
                f"something went wrong talking to the model: {result.error}*"
            )
        else:
            msg.content = (
                f"{ERROR_MARKER} ERROR! Sorry, something went wrong "
                f"talking to the model: {result.error}"
            )
        await msg.update()
    else:
        await msg.update()
        save_message_history(
            [
                *history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.text},
            ]
        )
        mark_faq_ids_used(retrieval.context_ids)
        await send_follow_up_suggestions(retrieval.suggestions)

    return result.response


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    # List available tools
    result: ListToolsResult = await session.list_tools()

    # Process tool metadata
    tools: list[FunctionToolParam] = [
        mcp_tool_to_openai_tool(tool) for tool in result.tools
    ]

    # Store tools for later use
    mcp_tools: dict[str, list[FunctionToolParam]] = cast(
        dict[str, list[FunctionToolParam]], cl.user_session.get("mcp_tools", {})
    )

    mcp_tools[str(connection.name)] = tools
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.step(type="tool")
async def call_tool(tool_use, input_items, msg) -> str:
    tool_name = tool_use["name"]
    tool_arguments = tool_use["arguments"]
    tool_call_id = tool_use["call_id"]
    logger.info(f"call_tool was called! {tool_name=} {tool_arguments=} {tool_call_id}")
    logger.info(f"The input items array: {input_items}")

    current_step = cl.context.current_step
    if current_step is not None:
        current_step.name = tool_name

    # Identify which mcp is used
    mcp_tools: dict[str, list[FunctionToolParam]] = cast(
        dict[str, list[FunctionToolParam]], cl.user_session.get("mcp_tools", {})
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
        logger.error(error_output)
        return error_output

    mcp_sessions = getattr(cl.context.session, "mcp_sessions", {}) or {}
    mcp_wrapper = mcp_sessions.get(mcp_name)

    if not mcp_wrapper:
        error_output = json.dumps(
            {"error": f"MCP {mcp_name} not found in any MCP connection"}
        )
        if current_step is not None:
            current_step.output = error_output
        logger.error(error_output)
        return error_output
    mcp_session = mcp_wrapper.client

    try:
        result = await mcp_session.call_tool(tool_name, tool_arguments)
        if current_step is not None:
            current_step.output = result
        logger.info(f"tool cal {result=}, text={result.content[0].text}")
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": result.content[0].text if isinstance(result.content[0].text, str) else json.dumps(result.structuredContent),
            }
        )
        logger.info("before LLM call for tool call response")
        await stream_assistant_reply(msg, input_items)
        logger.info("after LLM call for tool call response")

        return result
    except Exception as e:
        error_output = json.dumps({"error": str(e)})
        if current_step is not None:
            current_step.output = error_output
        logger.error(error_output)
        return error_output


@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])
    cl.user_session.set("active_follow_up_message", None)
    cl.user_session.set("used_faq_ids", set())


@cl.on_message
async def main(message: cl.Message):
    await answer_question(message.content)


@cl.action_callback(FOLLOW_UP_ACTION_NAME)
async def on_follow_up_click(action: cl.Action):
    question = action.payload.get("question", "")
    if question:
        await cl.Message(content=question, type="user_message").send()
        await answer_question(question)
