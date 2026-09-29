import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# ---------------------------------------------------------------------------
# Backend setup (same graph as your script, cached so it only builds once)
# ---------------------------------------------------------------------------

load_dotenv()


@st.cache_resource(show_spinner=False)
def build_app():
    set_llm_cache(SQLiteCache(database_path="femalebot.db"))

    groq_api_key = os.getenv("GROQ_API_KEY")

    search_tool = TavilySearch(max_results=1)
    tools = [search_tool]
    tool_node = ToolNode(tools=tools)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    llm_with_tool = llm.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    SYSTEM_PROMPT = """You are female friend bot designed to help men ask sensitive questions regarding women.
You have to guide the user who will be a man.
They will ask you questions regarding women, you have to answer as their woman friend.
Remember to keep the tone friendly and helpful.
Don't answer any erotic questions.
You will help the male user discuss some issues regarding women.
You can answer questions regarding sex education, but not erotica.
At the end of the answer, always display a random fact about women so it feels a little personal and friendly."""

    def chat(state: State):
        last_msg = state["messages"][-1].content
        response = llm_with_tool.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=last_msg),
            ]
        )
        return {"messages": response}

    graph = StateGraph(State)
    graph.add_node("chat", chat)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()


graph = build_app()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sakhi — Your Friendly Guide",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff7fb 0%, #ffffff 40%);
        color: #2b2b2b;
    }
    .stChatMessage, .stChatMessage p, .stChatMessage li, .stChatMessage span {
        border-radius: 14px;
        color: #2b2b2b !important;
    }
    h1 {color: #b5406b;}
    .subtitle {color: #7a7a7a; font-size: 0.95rem; margin-top: -10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 Sakhi")
st.markdown(
    '<p class="subtitle">A friendly space to ask honest questions about women — judgment-free.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "Sakhi answers everyday questions about women's health, feelings, "
        "and relationships in a warm, friend-to-friend tone. "
        "It won't engage with erotic content."
    )
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

avatars = {"user": "🧑", "assistant": "👩"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=avatars.get(msg["role"])):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatars["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=avatars["assistant"]):
        placeholder = st.empty()
        full_text = ""
        # stream_mode="messages" yields (message_chunk, metadata) pairs,
        # giving true token-by-token streaming instead of one final blob.
        for chunk, metadata in graph.stream(
            {"messages": prompt}, stream_mode="messages"
        ):
            if metadata.get("langgraph_node") != "chat":
                continue
            if isinstance(chunk, AIMessage) and chunk.content:
                full_text += chunk.content
                placeholder.markdown(full_text + "▌")

        placeholder.markdown(full_text or "_(no response — try rephrasing)_")

    st.session_state.messages.append({"role": "assistant", "content": full_text})