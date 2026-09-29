"""
03_RAG_Document_Chatbot_App.py
Category: Retrieval-Augmented Generation (RAG) Document Agent
Features:
- Free, local CPU embeddings via HuggingFaceEmbeddings (no OpenAI credits needed)
- In-memory FAISS document indexing for uploaded PDFs and TXTs
- Context-grounded conversation with LangGraph memory
"""

import os
import tempfile
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import streamlit as st

# 1. Environment & API Setup
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env file.")
    st.stop()

# 2. Local Free Embeddings (Runs locally on CPU)
@st.cache_resource
def get_embeddings():
    """Initializes a local sentence-transformer embedding model (zero API cost)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


embeddings = get_embeddings()

model = ChatOpenAI(
    model="openrouter/free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
)


# 3. RAG State Schema
class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


# 4. Streamlit File Upload & Indexing
st.set_page_config(page_title="RAG Document Chatbot", page_icon="📚", layout="wide")
st.title("📚 RAG Chatbot: Document Q&A with Memory")

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])


@st.cache_resource(show_spinner=False)
def index_document(file_bytes, filename):
    """Parses, splits, and indexes the document into an in-memory FAISS store."""
    ext = os.path.splitext(filename)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path) if ext == ".pdf" else TextLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


retriever = None
if uploaded_file:
    with st.spinner("Indexing document chunks into FAISS vector store..."):
        retriever = index_document(uploaded_file.getvalue(), uploaded_file.name)
    st.sidebar.success(f"Indexed: {uploaded_file.name}")
else:
    st.info("👈 Please upload a PDF or TXT document in the sidebar to start asking questions.")


# 5. Build RAG Graph
def retrieve_node(state: RAGState) -> dict:
    """Retrieves relevant document excerpts based on the user's latest query."""
    if not retriever:
        return {"context": "No document indexed."}
    latest_query = state["messages"][-1].content
    docs = retriever.invoke(latest_query)
    combined_context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": combined_context}


def generate_node(state: RAGState) -> dict:
    """Synthesizes a grounded answer using retrieved context and conversation history."""
    system_prompt = (
        "You are a helpful knowledge assistant. Use the following retrieved document context "
        "to answer the user question. If the answer cannot be found in the context, say so.\n\n"
        f"Context:\n{state.get('context', 'None')}"
    )
    messages = [HumanMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


@st.cache_resource
def get_rag_graph():
    checkpointer = MemorySaver()
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer)


rag_workflow = get_rag_graph()
config = {"configurable": {"thread_id": "rag-session"}}

# 6. Chat Interface
snapshot = rag_workflow.get_state(config)
for msg in snapshot.values.get("messages", []):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_q = st.chat_input("Ask a question about the document...")
if user_q:
    if not uploaded_file:
        st.warning("Please upload a document first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Searching document & formulating answer..."):
            output = rag_workflow.invoke(
                {"messages": [HumanMessage(content=user_q)]}, config=config
            )

        with st.expander("📖 Retrieved Document Excerpts"):
            st.write(output.get("context", "No context found."))

        st.markdown(output["messages"][-1].content)