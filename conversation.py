import re

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

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


@tool
def search_documents(query: str) -> str:
    """Search indexed local documents for information."""
    return document_engine.query_documents(query)


# Set description dynamically so the LLM knows when to invoke this tool
search_documents.description = (
    f"Search the local document library for information. "
    f"The documents relate to: {SYSTEM_DESCRIPTION} "
    f"Use this tool for any question the user asks that might be covered in the documents "
    f"rather than answered from general knowledge."
)


def get_config_values(config: RunnableConfig) -> RunnableConfig:
    metadata = config.get("metadata", {})
    config_values: RunnableConfig = {
        "configurable": {
            "user_id": metadata.get("user_id"),
            "thread_id": metadata.get("thread_id"),
        },
        "metadata": metadata
    }
    return config_values


def conversation(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    print(f"Latest message: {messages[-1].content if messages else ''}")

    # Pass the full accumulated history so the inner agent has conversation context
    inputs = {"messages": [("system", get_system_description())] + list(messages)}

    final_state = None

    for s in conversation_react_agent.stream(inputs, config=get_config_values(config), stream_mode="values"):
        final_state = s

        # Check for tool calls in the latest message
        if "messages" in s and s["messages"]:
            latest = s["messages"][-1]
            if hasattr(latest, 'tool_calls') and latest.tool_calls:
                print(f"Detected tool calls: {[tc.get('name', '') for tc in latest.tool_calls]}")

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


def ask_stuff(base_prompt: str, user_id: str, source: MessageSource) -> str:
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)  # Clean special characters
    full_prompt = format_prompt(base_prompt, source, user_id_clean)

    print(f"Role description: {get_system_description()}")
    print(f"Prompt to ask: {full_prompt}")

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
                print(f"Message tuple: {message}")
            elif hasattr(message, 'pretty_print'):
                message.pretty_print()

    final_text = ""
    if final_state and "messages" in final_state and final_state["messages"]:
        last_msg = final_state["messages"][-1]
        final_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    return final_text


ollama_instance = ChatOllama(model=THINKING_OLLAMA_MODEL)
conversation_tools = [search_documents]
conversation_react_agent = create_react_agent(ollama_instance, conversation_tools)

workflow = StateGraph(MessagesState)

workflow.add_edge(START, CONVERSATION_NODE)
workflow.add_node(CONVERSATION_NODE, conversation)
workflow.add_edge(CONVERSATION_NODE, END)

app = workflow.compile(checkpointer=MemorySaver())
