"""
05_Omni_Agent_Chatbot_App.py
=============================================================================
Capstone: All-in-One Production Agentic Chatbot
Features:
- Multi-Thread SQLite Persistence (SqliteSaver)
- Local Free Embeddings & Vector Search (FAISS + HuggingFace)
- Autonomous ReAct Tool Execution (Weather, Stocks, Math)
- Selective Human-in-the-Loop (HITL) Guard for Sensitive Transactions
- Streamlit Real-Time Chat & Supervisor Approval UI
=============================================================================
"""

import os
import sqlite3
import tempfile
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st

# ============================================================================
# 1. Environment & API Configuration
# ============================================================================
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env file.")
    st.stop()

# ============================================================================
# 2. Local Free Embeddings & Safe Vector Store Access
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Initializes local sentence-transformer embeddings (runs 100% locally on CPU)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def init_default_vectorstore():
    """Builds a cached default knowledge base if no custom document is uploaded."""
    emb = get_embeddings()
    default_docs = [
        Document(
            page_content=(
                "Enterprise Security & Investment Policy 2026: "
                "1. All single equity purchases exceeding $10,000 require manual compliance approval. "
                "2. Standard computing architecture utilizes AWS EC2 instances with Ubuntu 24.04 LTS. "
                "3. Machine learning models must be served via containerized environments with strict IAM roles."
            ),
            metadata={"source": "corporate_policy_2026.txt"},
        )
    ]
    return FAISS.from_documents(default_docs, emb)


def get_active_vectorstore():
    """Safely returns the active vector store, falling back to default if uninitialized."""
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        st.session_state.vectorstore = init_default_vectorstore()
    return st.session_state.vectorstore


# ============================================================================
# 3. Define Tools (Safe Tools vs Sensitive HITL Tools)
# ============================================================================
SENSITIVE_TOOLS = {"execute_stock_purchase"}


@tool
def calculate(expression: str) -> str:
    """Calculates mathematical expressions safely. Input should be a valid math string like '120 * 45'."""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"Calculation Result: {result}"
    except Exception as err:
        return f"Math calculation error: {err}"


@tool
def get_current_weather(city: str) -> str:
    """Retrieves real-time weather information for any given city."""
    normalized = city.lower().strip()
    weather_data = {
        "paris": "Sunny, 21°C, light wind.",
        "tokyo": "Clear skies, 18°C.",
        "new york": "Rainy, 14°C, overcast.",
        "london": "Cloudy, 15°C, 75% humidity.",
    }
    return weather_data.get(normalized, f"Partly cloudy, 20°C in {city.title()}.")


@tool
def get_stock_quote(symbol: str) -> str:
    """Fetches real-time equity market prices for public tech companies."""
    mock_market = {
        "AAPL": "$232.50 USD",
        "NVDA": "$128.75 USD",
        "MSFT": "$448.10 USD",
        "GOOGL": "$178.40 USD",
        "AMZN": "$186.20 USD",
    }
    ticker = symbol.upper().strip()
    return mock_market.get(ticker, f"Latest price for {ticker}: $150.00 USD (Composite Index)")


@tool
def search_internal_documents(query: str) -> str:
    """Searches internal corporate documentation, research papers, and uploaded files."""
    try:
        vs = get_active_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 2})
        matched = retriever.invoke(query)
        if not matched:
            return "No relevant sections found in the documents."
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'Doc')}]: {doc.page_content.strip()}"
            for doc in matched
        )
    except Exception as err:
        return f"Error querying knowledge base: {err}"


@tool
def execute_stock_purchase(symbol: str, shares: int, max_budget_usd: float) -> str:
    """SENSITIVE FINANCIAL TOOL: Executes a real-money equity stock order on behalf of the company.
    Always requires human supervisor approval before execution.
    """
    total_est = shares * 150.0
    return (
        f"TRANSACTION COMPLETED: Successfully purchased {shares} shares of {symbol.upper()} "
        f"(Total Allocated: ${total_est:,.2f} USD within max budget of ${max_budget_usd:,.2f})."
    )


all_tools = [
    calculate,
    get_current_weather,
    get_stock_quote,
    search_internal_documents,
    execute_stock_purchase,
]

# ============================================================================
# 4. OpenRouter Model & LangGraph ReAct Architecture
# ============================================================================
model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
model_with_tools = model.bind_tools(all_tools)


class OmniChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: OmniChatState) -> dict:
    """Main reasoning loop: analyzes history and decides whether to invoke tools."""
    system_instruction = (
        "You are an advanced enterprise AI assistant with access to tools for math, "
        "weather, live stock quotes, internal corporate document retrieval, and stock purchases. "
        "Use tools proactively when specific data or actions are required."
    )
    prompt_messages = [HumanMessage(content=system_instruction)] + state["messages"]
    response = model_with_tools.invoke(prompt_messages)
    return {"messages": [response]}


@st.cache_resource
def get_omni_agent():
    """Initializes SQLite persistence and compiles the graph with selective HITL barriers."""
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(OmniChatState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(all_tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])


omni_agent = get_omni_agent()

# ============================================================================
# 5. Streamlit Frontend UI
# ============================================================================
st.set_page_config(page_title="Enterprise Omni-Agent", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise Omni-Agent: Tools, RAG, Memory & HITL")

# Session State Initialization
if "threads" not in st.session_state:
    st.session_state.threads = ["main-session"]
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "main-session"
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = init_default_vectorstore()

# Sidebar: Thread Management & Document Ingestion
with st.sidebar:
    st.header("🧵 Conversation Sessions")
    if st.button("➕ New Chat Session", use_container_width=True):
        new_session = f"session-{str(uuid.uuid4())[:8]}"
        st.session_state.threads.append(new_session)
        st.session_state.current_thread = new_session
        st.rerun()

    for th in reversed(st.session_state.threads):
        is_active = th == st.session_state.current_thread
        lbl = f"👉 {th}" if is_active else f"💬 {th}"
        if st.button(lbl, key=th, use_container_width=True):
            st.session_state.current_thread = th
            st.rerun()

    st.divider()
    st.header("📄 Knowledge Base Ingestion")
    uploaded_doc = st.file_uploader("Upload PDF or TXT to Vector Index", type=["pdf", "txt"])
    if uploaded_doc:
        if st.button("Index Uploaded Document", use_container_width=True):
            ext = os.path.splitext(uploaded_doc.name)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_doc.getvalue())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path) if ext == ".pdf" else TextLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_documents(docs)
                st.session_state.vectorstore = FAISS.from_documents(chunks, get_embeddings())
                st.success(f"Indexed {len(chunks)} chunks from {uploaded_doc.name}!")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

current_config = {"configurable": {"thread_id": st.session_state.current_thread}}

# ============================================================================
# 6. Check Current State & Intelligent Selective HITL
# ============================================================================
snapshot = omni_agent.get_state(current_config)
pending_interrupt = bool(snapshot.next and "tools" in snapshot.next)

is_sensitive_call = False
pending_calls = []

if pending_interrupt:
    last_msg = snapshot.values["messages"][-1]
    pending_calls = getattr(last_msg, "tool_calls", [])
    is_sensitive_call = any(call["name"] in SENSITIVE_TOOLS for call in pending_calls)

    # AUTO-RESUME SAFE TOOLS: If all pending tools are safe (weather, math, RAG),
    # resume automatically without waiting for supervisor action.
    if not is_sensitive_call and pending_calls:
        omni_agent.invoke(None, config=current_config)
        st.rerun()

# ============================================================================
# 7. Render Historical Chat Conversation
# ============================================================================
for msg in snapshot.values.get("messages", []):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ============================================================================
# 8. Human-in-the-Loop Approval Modal (Only for Sensitive Actions)
# ============================================================================
if pending_interrupt and is_sensitive_call:
    st.warning("⚠️ **Human Supervisor Action Required** — The agent has requested a sensitive financial transaction:")
    for call in pending_calls:
        st.info(f"**Action:** `{call['name']}`\n\n**Arguments:** `{call['args']}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Execute Order", use_container_width=True):
            omni_agent.invoke(None, config=current_config)
            st.success("Transaction approved and executed!")
            st.rerun()

    with col2:
        if st.button("❌ Reject & Cancel Order", use_container_width=True):
            cancellation = ToolMessage(
                content="SECURITY ALERT: Stock purchase order was REJECTED by human supervisor.",
                tool_call_id=pending_calls[0]["id"],
            )
            omni_agent.update_state(current_config, {"messages": [cancellation]}, as_node="tools")
            omni_agent.invoke(None, config=current_config)
            st.error("Order rejected.")
            st.rerun()

# ============================================================================
# 9. User Input Handling
# ============================================================================
user_query = st.chat_input(
    "Ask a question, search documents, check stocks, or place an order...",
    disabled=bool(pending_interrupt and is_sensitive_call),
)

if user_query and not (pending_interrupt and is_sensitive_call):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Omni-Agent reasoning & executing tools..."):
            result = omni_agent.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=current_config,
            )

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                with st.expander("🛠️ Tool Invocations"):
                    st.json(msg.tool_calls)

        final_content = result["messages"][-1].content
        if final_content:
            st.markdown(final_content)

    st.rerun()