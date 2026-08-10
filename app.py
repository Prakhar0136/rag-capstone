import streamlit as st
import qdrant_client
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise AI Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #0b0f19;
        color: #f1f5f9;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }


    /* ---------- HEADER ---------- */

    .hero {
        padding: 10px 0 25px 0;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 20px;
    }


    /* ---------- STATUS ---------- */

    .status-container {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.25);
        padding: 7px 13px;
        border-radius: 999px;
        font-size: 13px;
        color: #86efac;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.7);
    }


    /* ---------- CHAT ---------- */

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.10);
    }


    /* ---------- INPUT ---------- */

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    [data-testid="stChatInput"] textarea {
        background: #111827 !important;
        color: #f8fafc !important;
        border-radius: 16px !important;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #080c14;
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
    }

    .sidebar-description {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.5;
    }


    /* ---------- CARDS ---------- */

    .info-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .info-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
    }

    .info-value {
        font-size: 14px;
        color: #e2e8f0;
        margin-top: 4px;
    }


    /* ---------- SOURCE CARD ---------- */

    .source-card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
        margin: 7px 0;
    }

    .source-title {
        font-weight: 600;
        font-size: 13px;
        color: #e2e8f0;
    }

    .source-score {
        font-size: 11px;
        color: #64748b;
        margin-top: 3px;
    }


    /* ---------- EMPTY STATE ---------- */

    .empty-state {
        text-align: center;
        padding: 70px 20px 50px 20px;
    }

    .empty-icon {
        font-size: 52px;
        margin-bottom: 12px;
    }

    .empty-title {
        font-size: 24px;
        font-weight: 650;
        color: #f8fafc;
    }

    .empty-text {
        color: #64748b;
        font-size: 14px;
        max-width: 500px;
        margin: auto;
    }


    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# QUERY ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_query_engine():

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    llm = Groq(
        model="openai/gpt-oss-20b"
    )

    client = qdrant_client.QdrantClient(
        url="http://localhost:6333"
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="capstone_docs"
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    qa_prompt_tmpl_str = (
        "You are an enterprise customer support assistant.\n\n"
        "Answer the user's question USING ONLY the provided context.\n"
        "Do not use outside knowledge.\n"
        "If the context does not contain enough information, say "
        "'I don't have enough information in the knowledge base to answer that.'\n\n"
        "---------------------\n"
        "Context:\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
        "Answer:"
    )

    qa_prompt = PromptTemplate(qa_prompt_tmpl_str)

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        text_qa_template=qa_prompt
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your **Customer Support Knowledge Assistant**.\n\n"
                "Ask me anything about the documents, policies, manuals, "
                "or knowledge base."
            )
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 Support AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Enterprise knowledge assistant powered by RAG.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ System")

    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">LLM</div>
            <div class="info-value">GPT-OSS 20B via Groq</div>
        </div>

        <div class="info-card">
            <div class="info-label">Embedding Model</div>
            <div class="info-value">BGE Small EN v1.5</div>
        </div>

        <div class="info-card">
            <div class="info-label">Vector Database</div>
            <div class="info-value">Qdrant</div>
        </div>

        <div class="info-card">
            <div class="info-label">Retrieval</div>
            <div class="info-value">Top 3 semantic chunks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 💡 Example Questions")

    example_questions = [
        "What is the refund policy?",
        "How can I reset my password?",
        "What are the product warranty terms?",
        "What should I do if my order is delayed?"
    ]

    for question in example_questions:

        if st.button(
            question,
            use_container_width=True,
            key=f"example_{question}"
        ):
            st.session_state.pending_question = question
            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. 👋\n\n"
                    "How can I help you?"
                )
            }
        ]

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.html("""
    <div class="hero">

        <div class="hero-title">
            🤖 Customer Support Knowledge Bot
        </div>

        <div class="hero-subtitle">
            Ask questions about company policies, product manuals,
            and internal documentation.
        </div>

        <div class="status-container">
            <span class="status-dot"></span>
            Knowledge Base Connected
        </div>

    </div>
""")


# ============================================================
# LOAD ENGINE
# ============================================================

with st.spinner("Connecting to knowledge base..."):

    try:

        query_engine = load_query_engine()

    except Exception as e:

        st.error("❌ Unable to initialize the AI system.")

        with st.expander("Technical details"):

            st.exception(e)

        st.stop()


# ============================================================
# EMPTY STATE
# ============================================================

if len(st.session_state.messages) == 1:

    st.html("""
        <div class="empty-state">

            <div class="empty-icon">📚</div>

            <div class="empty-title">
                Ask your knowledge base
            </div>

            <div class="empty-text">
                Search through your company documentation using
                natural language. Answers are grounded in your
                indexed documents.
            </div>

        </div>
""")


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"],
        avatar="🤖" if msg["role"] == "assistant" else "👤"
    ):

        st.markdown(msg["content"])

        # Show retrieved sources if they exist
        if msg["role"] == "assistant" and "sources" in msg:

            with st.expander(
                f"🔍 Retrieved Sources ({len(msg['sources'])})"
            ):

                for idx, source in enumerate(
                    msg["sources"],
                    1
                ):

                    score = source.score

                    st.html(f"""
                        <div class="source-card">

                            <div class="source-title">
                                📄 Source {idx}
                            </div>

                            <div class="source-score">
                                Similarity score:
                                {score:.4f}
                            </div>

                        </div>
""")

                    st.text(
                        source.node.get_content()
                    )


# ============================================================
# INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something about your documents..."
)


# Handle sidebar example question
if (
    "pending_question" in st.session_state
    and not user_input
):

    user_input = st.session_state.pending_question

    del st.session_state.pending_question


# ============================================================
# PROCESS QUERY
# ============================================================

if user_input:

    # -------------------------
    # User message
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(user_input)


    # -------------------------
    # Assistant response
    # -------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🔎 Searching knowledge base..."
        ):

            try:

                response = query_engine.query(
                    user_input
                )

                answer = response.response

                st.markdown(answer)


                # -------------------------
                # Retrieval metadata
                # -------------------------

                sources = response.source_nodes

                if sources:

                    st.divider()

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Sources Retrieved",
                            len(sources)
                        )

                    with col2:

                        scores = [
                            s.score
                            for s in sources
                            if s.score is not None
                        ]

                        if scores:

                            avg_score = (
                                sum(scores) / len(scores)
                            )

                            st.metric(
                                "Avg. Similarity",
                                f"{avg_score:.3f}"
                            )


                    with st.expander(
                        f"🔍 View Retrieved Sources ({len(sources)})"
                    ):

                        for idx, source in enumerate(
                            sources,
                            1
                        ):

                            score = source.score

                            st.html(f"""
                                <div class="source-card">

                                    <div class="source-title">
                                        📄 Retrieved Chunk {idx}
                                    </div>

                                    <div class="source-score">
                                        Similarity:
                                        {score:.4f}
                                    </div>

                                </div>
""")

                            st.text(
                                source.node.get_content()
                            )


                    # Store sources in history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                else:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


            except Exception as e:

                answer = (
                    "Sorry, I encountered an error while "
                    "searching the knowledge base."
                )

                st.error(answer)

                with st.expander(
                    "🔧 Technical details"
                ):

                    st.exception(e)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#475569;
        font-size:12px;
        margin-top:40px;
        padding-bottom:20px;
    ">
        Enterprise RAG Support Assistant •
        Qdrant + LlamaIndex + Groq
    </div>
    """,
    unsafe_allow_html=True
)
