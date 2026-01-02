import os
import re
import ast
import operator as op
from typing import TypedDict, List, Literal, Optional, Dict

import streamlit as st

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Optional LLM (only used if OPENAI_API_KEY is set)
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


# -----------------------------
# Small "knowledge base" (RAG-lite)
# -----------------------------
KB = [
    {
        "title": "Month-end close (why it hurts)",
        "text": (
            "Month-end close is the process of finalizing financial records for a period. "
            "Common pain points: manual reconciliations, spreadsheet dependency, journal entry volume, "
            "late adjustments, and limited audit trails. Automation reduces close time and errors."
        ),
    },
    {
        "title": "Reconciliations (what good looks like)",
        "text": (
            "Reconciliation verifies that two sources of truth match (e.g., bank vs ledger). "
            "Best practices: clear matching rules, exception queues, audit logs, and repeatable workflows."
        ),
    },
    {
        "title": "Journal entries (automation ideas)",
        "text": (
            "Journal entries record financial transactions. Automation can generate recurring entries, "
            "validate account mappings, enforce approval workflows, and reduce manual posting mistakes."
        ),
    },
    {
        "title": "RAG (Retrieval-Augmented Generation)",
        "text": (
            "RAG improves LLM answers by retrieving relevant context from a knowledge base. "
            "Flow: user query -> retrieval -> LLM uses retrieved context -> grounded answer."
        ),
    },
]


def simple_retrieve(query: str, k: int = 2) -> str:
    """Very small retrieval: keyword overlap scoring (no embeddings needed)."""
    q = set(re.findall(r"[a-zA-Z]+", query.lower()))
    scored = []
    for doc in KB:
        d = set(re.findall(r"[a-zA-Z]+", (doc["title"] + " " + doc["text"]).lower()))
        score = len(q.intersection(d))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for s, d in scored[:k] if s > 0]
    if not top:
        return ""
    return "\n\n".join([f"### {d['title']}\n{d['text']}" for d in top])


# -----------------------------
# Safe calculator tool
# -----------------------------
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if isinstance(node, ast.Constant):  # py>=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


def calc_from_text(text: str) -> Optional[float]:
    """
    Extract a simple arithmetic expression from text and evaluate safely.
    Supports: + - * / ** % and parentheses.
    """
    # Grab the "math-looking" part
    m = re.findall(r"[\d\.\(\)\+\-\*\/\%\s]+", text)
    if not m:
        return None
    expr = max(m, key=len).strip()
    # Basic sanity
    if not any(ch.isdigit() for ch in expr):
        return None
    try:
        parsed = ast.parse(expr, mode="eval")
        return float(_safe_eval(parsed.body))
    except Exception:
        return None


# -----------------------------
# LangGraph State
# -----------------------------
Route = Literal["retrieve", "calc", "direct"]

class GraphState(TypedDict):
    messages: List[BaseMessage]
    route: Route
    context: str
    calc_result: str


def get_llm():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or ChatOpenAI is None:
        return None
    # Keep it cheap/fast; adjust as you like
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# -----------------------------
# LangGraph Nodes
# -----------------------------
def node_route(state: GraphState) -> GraphState:
    user_msg = state["messages"][-1].content

    llm = get_llm()
    if llm:
        prompt = (
            "You are a router. Choose ONE route for the user's message:\n"
            "- 'calc' if the user asks to compute math\n"
            "- 'retrieve' if the user asks about month-end close, reconciliations, journal entries, NetSuite, RAG\n"
            "- 'direct' otherwise\n\n"
            "Return ONLY one word: calc | retrieve | direct\n\n"
            f"User: {user_msg}"
        )
        out = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()
        route: Route = "direct"
        if "calc" in out:
            route = "calc"
        elif "retrieve" in out:
            route = "retrieve"
        else:
            route = "direct"
    else:
        # Rule-based fallback router
        if calc_from_text(user_msg) is not None:
            route = "calc"
        elif any(k in user_msg.lower() for k in ["close", "recon", "journal", "netsuite", "rag", "audit"]):
            route = "retrieve"
        else:
            route = "direct"

    state["route"] = route
    return state


def node_retrieve(state: GraphState) -> GraphState:
    user_msg = state["messages"][-1].content
    context = simple_retrieve(user_msg, k=2)
    state["context"] = context
    return state


def node_calc(state: GraphState) -> GraphState:
    user_msg = state["messages"][-1].content
    val = calc_from_text(user_msg)
    state["calc_result"] = "" if val is None else str(val)
    return state


def node_respond(state: GraphState) -> GraphState:
    user_msg = state["messages"][-1].content
    context = state.get("context", "")
    calc_result = state.get("calc_result", "")

    llm = get_llm()
    if llm:
        system = (
            "You are a helpful AI engineer assistant. Answer clearly with short sections.\n"
            "If context is provided, use it and do not hallucinate beyond it.\n"
            "If calc_result is provided, present it clearly.\n"
        )
        prompt = f"""
User question:
{user_msg}

Context (may be empty):
{context}

Calc result (may be empty):
{calc_result}

Write the best helpful answer.
"""
        answer = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)]).content
    else:
        # No-LLM fallback: template response
        if calc_result:
            answer = f"**Result:** {calc_result}\n\n(Enable `OPENAI_API_KEY` for richer explanations.)"
        elif context:
            answer = f"Here’s what I found in the local KB:\n\n{context}\n\n(Enable `OPENAI_API_KEY` for a more polished answer.)"
        else:
            answer = "I can help! (Enable `OPENAI_API_KEY` for an LLM answer, or ask about close/recon/journals/RAG.)"

    state["messages"].append(AIMessage(content=answer))
    return state


# -----------------------------
# Build LangGraph
# -----------------------------
@st.cache_resource
def build_graph():
    g = StateGraph(GraphState)

    g.add_node("route", node_route)
    g.add_node("retrieve", node_retrieve)
    g.add_node("calc", node_calc)
    g.add_node("respond", node_respond)

    g.set_entry_point("route")

    def pick_next(state: GraphState) -> str:
        return state["route"]

    g.add_conditional_edges(
        "route",
        pick_next,
        {
            "retrieve": "retrieve",
            "calc": "calc",
            "direct": "respond",
        },
    )

    g.add_edge("retrieve", "respond")
    g.add_edge("calc", "respond")
    g.add_edge("respond", END)

    return g.compile()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Streamlit Demo", page_icon="🕸️", layout="wide")
st.title("🕸️ LangGraph + Streamlit Demo")
st.caption("Router graph: retrieve (RAG-lite) • calc tool • direct answer. Add OPENAI_API_KEY for LLM.")

graph = build_graph()

if "chat" not in st.session_state:
    st.session_state.chat = [
        AIMessage(content="Ask me about month-end close, reconciliations, journal entries, or try a calculation like (12*7)+5.")
    ]

with st.sidebar:
    st.subheader("Options")
    debug = st.toggle("Show debug (route/context)", value=True)
    st.divider()
    st.write("**Try prompts:**")
    st.write("- What is month-end close and why is it painful?")
    st.write("- Explain reconciliation best practices")
    st.write("- Calculate (1450*0.18)+99")
    st.write("- What is RAG?")

# Render chat
for msg in st.session_state.chat:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Type your message…")

if user_input:
    st.session_state.chat.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build state for LangGraph
    state: GraphState = {
        "messages": st.session_state.chat.copy(),
        "route": "direct",
        "context": "",
        "calc_result": "",
    }

    # Run graph
    final_state = graph.invoke(state)

    # Persist assistant answer into session
    st.session_state.chat = final_state["messages"]

    with st.chat_message("assistant"):
        st.markdown(final_state["messages"][-1].content)

        if debug:
            with st.expander("Debug details", expanded=False):
                st.write("**Route chosen:**", final_state.get("route"))
                if final_state.get("calc_result"):
                    st.write("**Calc result:**", final_state.get("calc_result"))
                if final_state.get("context"):
                    st.markdown("**Retrieved context:**")
                    st.markdown(final_state.get("context"))
