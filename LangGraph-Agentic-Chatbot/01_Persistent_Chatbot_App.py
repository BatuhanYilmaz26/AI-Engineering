"""
01_Persistent_Chatbot_App.py
Category: Stateful Multi-Session Chatbot with SQLite Persistence
Features:
- LangGraph SqliteSaver checkpointer for permanent disk persistence
- Multi-thread chat sidebar (create new chats, switch between sessions)
- Real-time token streaming with st.write_stream
"""

import os
import sqlite3
import uuid
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import streamlit as st

# 1. Environment & API Setup
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env file.")
    st.stop()

# 2. OpenRouter Model Initialization
model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    streaming=True,
)


# 3. State Schema with Built-in Message Reducer
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict:
    """Executes the LLM turn and emits the assistant response delta."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# 4. Compile Workflow with SQLite Persistence
@st.cache_resource
def get_compiled_chatbot():
    """Initializes SQLite connection and compiles the LangGraph checkpointer."""
    # SQLite connection with check_same_thread=False for Streamlit concurrency
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)

    return builder.compile(checkpointer=checkpointer)


chatbot = get_compiled_chatbot()

# 5. Streamlit Frontend Layout
st.set_page_config(page_title="AI Agentic Chatbot", page_icon="💬", layout="wide")
st.title("💬 Agentic Chatbot: Permanent Memory & Threads")

# 6. Session & Thread Management
if "threads" not in st.session_state:
    st.session_state.threads = ["default-session"]
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default-session"

# Sidebar: Thread Management
with st.sidebar:
    st.header("🧵 Chat Threads")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = f"session-{str(uuid.uuid4())[:8]}"
        st.session_state.threads.append(new_id)
        st.session_state.current_thread = new_id
        st.rerun()

    st.caption("Active Sessions:")
    for thread in reversed(st.session_state.threads):
        is_active = thread == st.session_state.current_thread
        btn_label = f"👉 {thread}" if is_active else f"💬 {thread}"
        if st.button(btn_label, key=thread, use_container_width=True):
            st.session_state.current_thread = thread
            st.rerun()

current_thread_id = st.session_state.current_thread
thread_config = {"configurable": {"thread_id": current_thread_id}}

# 7. Render Historical Messages from LangGraph State Snapshot
snapshot = chatbot.get_state(thread_config)
persisted_messages = snapshot.values.get("messages", [])

for msg in persisted_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 8. User Input & Streaming Execution
user_query = st.chat_input("Type your message here...")
if user_query:
    # Display user input immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Stream the assistant response via LangGraph event stream
    with st.chat_message("assistant"):

        def response_stream():
            events = chatbot.stream(
                {"messages": [HumanMessage(content=user_query)]},
                config=thread_config,
                stream_mode="messages",
            )
            for msg_chunk, _ in events:
                if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                    yield msg_chunk.content

        st.write_stream(response_stream())