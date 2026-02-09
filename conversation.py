import re

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import InMemoryStore
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

import document_engine
from lore_utils import MessageSource, SYSTEM_DESCRIPTION, THINKING_OLLAMA_MODEL

CONVERSATION_NODE = "conversation"


def get_system_description():
    """
    Format the chatbot's system role description dynamically by including tools from the list.
    """
    return f"""
Role:
    {SYSTEM_DESCRIPTION}

Tools:
    search_documents: Search local documents. Use this for questions about anything you are not aware of.
    """


@tool(parse_docstring=True)
def search_documents(query: str):
    """
    Use this tool to get information when the user asks about the description provided in the prompt.

    Args:
    query: The question or search term to look for in the documents that match the search_documents description.

    Returns:
    string: The answer derived from the documents.
    """
    return document_engine.query_documents(query)


def get_config_values(config: RunnableConfig) -> RunnableConfig:
    metadata = config.get("metadata", {})
    config_values: RunnableConfig = {
        "configurable": {
            "user_id": metadata.get("user_id"),
            "thread_id": metadata.get("thread_id"),
        },
        "metadata": metadata  # Preserve full metadata including progress_callback
    }
    return config_values


def conversation(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    print("Latest message: %s", latest_message)
    inputs = {"messages": [("system", get_system_description()),
                           ("user", latest_message)]}

    final_state = None

    for s in conversation_react_agent.stream(inputs, config=get_config_values(config), stream_mode="values"):
        final_state = s

        # Check for tool calls in the latest message
        if "messages" in s and s["messages"]:
            latest = s["messages"][-1]
            if hasattr(latest, 'tool_calls') and latest.tool_calls:
                print("Detected tool calls: %s", [tc.get('name', '') for tc in latest.tool_calls])

    resp = final_state["messages"][-1].content if final_state and "messages" in final_state else ""
    return {'messages': [resp]}


def get_source_info(source: MessageSource, user_id: str) -> str:
    """Generate source information based on the messaging platform."""
    if source == MessageSource.DISCORD_TEXT:
        return f"User is texting from Discord (User ID: {user_id})"
    elif source == MessageSource.DISCORD_VOICE:
        return f"User is speaking from Discord (User ID: {user_id}). Please answer in 30 words or less."
    return f"User is interacting via CLI (User ID: {user_id})"


def format_prompt(prompt: str, source: MessageSource, user_id: str) -> str:
    """Format the final prompt for the chatbot."""
    return f"""
    Context:
        {get_source_info(source, user_id)}
    Question:
        {prompt}
    """


def ask_stuff(base_prompt: str, user_id: str, source: MessageSource) -> dict:
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)  # Clean special characters
    full_prompt = format_prompt(base_prompt, source, user_id_clean)

    system_prompt = get_system_description()
    print("Role description: %s", system_prompt)
    print("Prompt to ask: %s", full_prompt)

    config = {
        "configurable": {"user_id": user_id_clean, "thread_id": user_id_clean}
    }
    inputs = {"messages": [("user", full_prompt)]}

    # Collect final state from the stream
    final_state = None
    for s in app.stream(inputs, config=config, stream_mode="values"):
        final_state = s
        message = s["messages"][-1] if "messages" in s and s["messages"] else None
        if message:
            if isinstance(message, tuple):
                print("Message tuple: %s", message)
            elif hasattr(message, 'pretty_print'):
                message.pretty_print()

    final_text = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_msg = final_state["messages"][-1]
        final_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    return final_text


ollama_instance = ChatOllama(model=THINKING_OLLAMA_MODEL)
conversation_tools = [search_documents]
conversation_react_agent = create_agent(ollama_instance, tools=conversation_tools)

workflow = StateGraph(MessagesState)

workflow.add_edge(START, CONVERSATION_NODE)
workflow.add_node(CONVERSATION_NODE, conversation)
workflow.add_edge(CONVERSATION_NODE, END)

app = workflow.compile(checkpointer=MemorySaver(), store=InMemoryStore())
