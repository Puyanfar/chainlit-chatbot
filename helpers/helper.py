from openai.types.responses import EasyInputMessageParam, FunctionToolParam
from mcp.types import Tool
import chainlit as cl
import config

MAX_HISTORY_MESSAGES: int = config.MAX_HISTORY_MESSAGES
ASSISTANT_AUTHOR: str = "assistant"
FOLLOW_UP_ACTION_NAME: str = "follow_up_question"


def get_message_history() -> list[EasyInputMessageParam]:
    return cl.user_session.get("message_history") or []


def save_message_history(history: list[EasyInputMessageParam]) -> None:
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    cl.user_session.set("message_history", history)


def get_used_faq_ids() -> set[str]:
    return cl.user_session.get("used_faq_ids") or set()


def mark_faq_ids_used(context_ids: list[str]) -> None:
    if not context_ids:
        return
    used = get_used_faq_ids()
    cl.user_session.set("used_faq_ids", used | set(context_ids))


def build_follow_up_actions(suggestions: list[dict]) -> list[cl.Action]:
    """One cl.Action per suggestion, all sharing the same action name - each carries its own
    question text in its payload so the callback knows which was clicked."""
    return [
        cl.Action(
            name=FOLLOW_UP_ACTION_NAME,
            payload={"question": suggestion["question"]},
            label=suggestion["question"],
        )
        for suggestion in suggestions
    ]


async def send_follow_up_suggestions(suggestions: list) -> None:
    if not suggestions:
        return
    actions = build_follow_up_actions(suggestions)
    follow_up_message = cl.Message(content="", actions=actions, author=ASSISTANT_AUTHOR)
    await follow_up_message.send()
    cl.user_session.set("active_follow_up_message", follow_up_message)


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


def mcp_tool_to_openai_tool(mcp_tool: Tool) -> FunctionToolParam:
    return {
        "type": "function",
        "name": mcp_tool.name,
        "description": mcp_tool.description or "",
        "parameters": mcp_tool.inputSchema,
        "strict": False,
    }
