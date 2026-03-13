

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from services.memory_service import get_recent_history, save_message
from rag.faiss_store import retrieve_chunks
from rag.ollama_client import generate_response


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class ChatState(TypedDict):
    """
    This is the shared state object passed between all nodes.
    Each field is populated by a different node in the pipeline.
    """
    thread_id: str        # unique ID for the conversation
    user_input: str       # the user's latest question
    retrieved_docs: list  # chunks fetched from FAISS
    history: list         # recent messages from SQLite
    answer: str           # final answer from OLLama


# ─────────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────────

def load_history_node(state: ChatState) -> dict:
    """
    NODE 1: Load Conversation History

    Fetches recent messages from SQLite for this thread.
    This gives the LLM context about what was discussed before.
    """
    thread_id = state["thread_id"]
    history = get_recent_history(thread_id)
    print(f" Loaded {len(history)} history messages for thread: {thread_id}")
    return {"history": history}


def retrieve_docs_node(state: ChatState) -> dict:
    """
    NODE 2: Retrieve Relevant Document Chunks

    Takes the user's question, embeds it, and searches FAISS
    for the most similar stored chunks.

    Returns empty list if no documents have been ingested yet.
    """
    user_input = state["user_input"]
    docs = retrieve_chunks(user_input)
    print(f" Retrieved {len(docs)} document chunks")
    return {"retrieved_docs": docs}


def build_prompt_node(state: ChatState) -> dict:
    """
    NODE 3: Build the Full Prompt

    Combines:
    - System instruction
    - Retrieved document context
    - Conversation history
    - User's question

    WHY THIS FORMAT:
    LLMs perform better when given clear structure.
    The "context" section helps ground answers in your documents.
    The "history" section maintains conversational flow.
    """
    # Format retrieved docs as numbered list
    context = "\n\n".join(
        [f"[{i+1}] {doc}" for i, doc in enumerate(state["retrieved_docs"])]
    ) or "No relevant documents found."

    # Format history as Human/Assistant turns
    history_text = ""
    for msg in state["history"]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    if not history_text:
        history_text = "No previous conversation."

    # Build the full prompt
    prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately. 
If the context doesn't contain the answer, say so honestly. Be concise and clear.

=== CONTEXT (from documents) ===
{context}

=== CONVERSATION HISTORY ===
{history_text}

=== CURRENT QUESTION ===
Human: {state["user_input"]}

Answer:"""

    # We store the prompt in the state but don't add a new key
    # Instead we temporarily use "answer" to pass the prompt to generate_node
    # (We overwrite it with the real answer in the next step)
    return {"answer": prompt}  # temp: store prompt here


def generate_node(state: ChatState) -> dict:
    """
    NODE 4: Generate Answer using OLLama

    Sends the full prompt to OLLama and gets back the answer.
    The "answer" field in state was temporarily storing the prompt —
    now we replace it with the actual generated answer.
    """
    prompt = state["answer"]  # this was set by build_prompt_node
    answer = generate_response(prompt)
    print(f" Generated answer ({len(answer)} chars)")
    return {"answer": answer}


def save_message_node(state: ChatState) -> dict:
    """
    NODE 5: Save Messages to SQLite

    Saves both the user's message and the assistant's answer
    so they appear in future history loads.
    """
    save_message(state["thread_id"], "user", state["user_input"])
    save_message(state["thread_id"], "assistant", state["answer"])
    print(f" Saved conversation to thread: {state['thread_id']}")
    return {}  # no state changes needed


# ─────────────────────────────────────────────
# Build the LangGraph Pipeline
# ─────────────────────────────────────────────

def build_chat_graph():
    """
    Assembles all nodes into a sequential pipeline.

    Flow:
    START → load_history → retrieve_docs → build_prompt → generate → save_message → END
    """
    # Create a new graph that uses ChatState as its state type
    graph = StateGraph(ChatState)

    # Add each node (name, function)
    graph.add_node("load_history", load_history_node)
    graph.add_node("retrieve_docs", retrieve_docs_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("generate", generate_node)
    graph.add_node("save_message", save_message_node)

    # Connect nodes in order (edges define the flow)
    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "retrieve_docs")
    graph.add_edge("retrieve_docs", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", "save_message")
    graph.add_edge("save_message", END)

    # Compile = finalize and validate the graph
    return graph.compile()


# Create the compiled graph (reused across requests)
chat_graph = build_chat_graph()


def run_chat(thread_id: str, user_input: str) -> str:
    """
    Main entry point to run the full chat pipeline.

    Args:
        thread_id: identifies the conversation
        user_input: the user's question

    Returns:
        The assistant's answer
    """
    # Initial state — nodes will fill in the other fields
    initial_state: ChatState = {
        "thread_id": thread_id,
        "user_input": user_input,
        "retrieved_docs": [],
        "history": [],
        "answer": "",
    }

    # Run the graph
    final_state = chat_graph.invoke(initial_state)

    return final_state["answer"]
