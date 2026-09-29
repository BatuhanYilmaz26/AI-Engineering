"""
04_HITL_Approval_Agent_App.py
Category: Human-in-the-Loop (HITL) Guarded Action Agent
Features:
- interrupt_before=["tools"] execution barrier
- Sensitive tool actions (Bank Wire Transfer, Account Balance)
- Streamlit Human Approval / Rejection interface to safely resume graph state
"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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


# 2. Define Safe and Sensitive Tools
@tool
def check_balance(account_id: str) -> str:
    """Safe Tool: Reads current customer balance without making changes."""
    return f"Account {account_id} balance: $12,450.00 USD"


@tool
def execute_wire_transfer(recipient: str, amount: float) -> str:
    """CRITICAL SENSITIVE TOOL: Dispatches an irreversible real-money wire transfer."""
    return f"TRANSACTION CONFIRMED: Successfully transferred ${amount:,.2f} to {recipient}."


tools = [check_balance, execute_wire_transfer]

# 3. Model Binding
model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
)
model_with_tools = model.bind_tools(tools)


# 4. Graph Construction with HITL Interrupt
class HITLState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: HITLState) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}


@st.cache_resource
def get_hitl_agent():
    checkpointer = MemorySaver()
    builder = StateGraph(HITLState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    # CRITICAL: Interrupt graph execution before entering the "tools" node
    return builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])


hitl_agent = get_hitl_agent()

# 5. Streamlit Frontend
st.set_page_config(page_title="HITL Agent", page_icon="🛡️", layout="wide")
st.title("🛡️ Human-in-the-Loop (HITL) Guarded Agent")
st.caption(
    "Safe action: `check_balance` | **Sensitive action requires human approval:** `execute_wire_transfer`"
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "hitl-session-1"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 6. Check Current Graph State for Pending Approvals
state = hitl_agent.get_state(config)
pending_action = state.next and "tools" in state.next

# Render Chat History
for msg in state.values.get("messages", []):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 7. Human-in-the-Loop Intercept Panel
if pending_action:
    # Retrieve proposed tool calls from the latest AI message
    last_ai_msg = state.values["messages"][-1]
    tool_calls = getattr(last_ai_msg, "tool_calls", [])

    st.warning("⚠️ **Human Approval Required** — The agent has requested a sensitive action:")
    for call in tool_calls:
        st.info(f"**Action:** `{call['name']}`\n\n**Arguments:** `{call['args']}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Execute", use_container_width=True):
            # Passing None resumes execution past the interrupt
            resumed_state = hitl_agent.invoke(None, config=config)
            st.success("Action approved and completed!")
            st.rerun()

    with col2:
        if st.button("❌ Reject & Abort", use_container_width=True):
            # Manually inject a tool cancellation message into state to inform the agent
            rejection_msg = ToolMessage(
                content="Action rejected by human supervisor.",
                tool_call_id=tool_calls[0]["id"] if tool_calls else "manual_reject",
            )
            hitl_agent.update_state(config, {"messages": [rejection_msg]}, as_node="tools")
            # Resume agent to generate acknowledgement of rejection
            hitl_agent.invoke(None, config=config)
            st.error("Action was rejected.")
            st.rerun()

# 8. User Input (Disabled while approval is pending)
user_prompt = st.chat_input(
    "Ask a question or request a transfer (e.g. 'Transfer $500 to Alice')...",
    disabled=bool(pending_action),
)

if user_prompt and not pending_action:
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Processing agent instructions..."):
        hitl_agent.invoke({"messages": [HumanMessage(content=user_prompt)]}, config=config)

    st.rerun()