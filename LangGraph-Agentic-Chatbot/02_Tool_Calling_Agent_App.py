"""
02_Tool_Calling_Agent_App.py
Category: Autonomous ReAct Agent with Tool Calling
Features:
- LangGraph ToolNode and tools_condition routing loop
- Calculator, Weather, and Finance tools
- Expandable Streamlit execution tracing
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st

# 1. Environment & Setup
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env file.")
    st.stop()


# 2. Define Custom Agent Tools
@tool
def calculate(expression: str) -> str:
    """Calculates mathematical expressions safely. Input should be a valid math string like '145 * 24'."""
    try:
        # Safe math evaluation using restricted globals
        result = eval(expression, {"__builtins__": None}, {})
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_current_weather(location: str) -> str:
    """Fetches real-time weather information for any given city."""
    # Simulated weather tool response
    conditions = {"tokyo": "Sunny, 22°C", "new york": "Rainy, 14°C", "london": "Cloudy, 16°C"}
    loc_clean = location.lower().strip()
    return conditions.get(loc_clean, f"Partly cloudy, 20°C with light wind in {location}.")


@tool
def get_stock_price(ticker: str) -> str:
    """Retrieves current market stock prices for public companies."""
    mock_market = {"AAPL": "$230.50", "NVDA": "$125.80", "GOOGL": "$175.20", "MSFT": "$440.10"}
    price = mock_market.get(ticker.upper().strip(), "$150.00 (Market Index Average)")
    return f"Latest price for {ticker.upper()}: {price}"


tools = [calculate, get_current_weather, get_stock_price]

# 3. Model Binding
model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
)
model_with_tools = model.bind_tools(tools)


# 4. Graph Construction (ReAct Agent Pattern)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: AgentState) -> dict:
    """Reasoning node: decides whether to respond directly or invoke a tool."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


@st.cache_resource
def get_agent_workflow():
    checkpointer = MemorySaver()
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    # Dynamic routing: routes to 'tools' if tool calls are present, else END
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # Loop back to agent with tool observation

    return builder.compile(checkpointer=checkpointer)


agent_executor = get_agent_workflow()

# 5. Streamlit UI
st.set_page_config(page_title="ReAct Agent Tools", page_icon="🛠️", layout="wide")
st.title("🛠️ ReAct Agent with Tool Execution")
st.caption("Available Tools: `calculate`, `get_current_weather`, `get_stock_price`")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "agent-tools-session"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Render History
snapshot = agent_executor.get_state(config)
for msg in snapshot.values.get("messages", []):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle Input
query = st.chat_input("Ask a question (e.g. 'What is 45 * 89?' or 'Weather in Tokyo')...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning & checking tools..."):
            result = agent_executor.invoke(
                {"messages": [HumanMessage(content=query)]}, config=config
            )

        # Inspect tool calls in an expander
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                with st.expander("🔍 Tool Invocation Details"):
                    st.json(msg.tool_calls)

        # Final response
        final_answer = result["messages"][-1].content
        st.markdown(final_answer)