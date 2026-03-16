import streamlit as st
import requests
import config
from rag_pipeline import RAGPipeline

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin-bottom: 0; }
    .subtitle { font-size: 1rem; color: #9ca3af; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .answer-box {
        background: #1e2130;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #e5e7eb;
        font-size: 0.97rem;
        line-height: 1.6;
    }
    .context-item {
        background: #252840;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        color: #c4c9d9;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**LLM:** `{config.OLLAMA_MODEL}` via Ollama")
    st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**Vector DB:** Endee @ `{config.ENDEE_BASE_URL}`")
    st.markdown("---")
    st.markdown("**Example queries:**")
    st.markdown("- Does the candidate have RAG experience?")
    st.markdown("- What vector databases has the candidate used?")
    st.markdown("- Does the candidate meet the job requirements?")
    st.markdown("- Summarise the skill gap between candidate and role.")


# ── Health checks ────────────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_endee() -> bool:
    try:
        r = requests.get(f"{config.ENDEE_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Load pipeline (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_pipeline():
    return RAGPipeline()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 AI Interview Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions about the candidate\'s resume and the job description — '
    "powered by <strong>Endee</strong> vector search + <strong>Mistral</strong> via Ollama.</p>",
    unsafe_allow_html=True,
)

# ── Status indicators ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    ollama_ok = check_ollama()
    if ollama_ok:
        st.success("Ollama: Running ✅")
    else:
        st.error("Ollama: Not reachable ❌")

with col2:
    endee_ok = check_endee()
    if endee_ok:
        st.success("Endee DB: Running ✅")
    else:
        st.warning("Endee DB: Not reachable ⚠️")

st.markdown("---")

# ── Main interface ───────────────────────────────────────────────────────────
if not ollama_ok:
    st.info(
        "**Ollama is not running.** Start it with:\n\n"
        "```bash\n"
        "ollama serve\n"
        "ollama pull mistral\n"
        "```"
    )
else:
    pipeline = load_pipeline()

    query = st.text_input(
        "Ask a question:",
        placeholder="e.g. Does the candidate have production RAG experience?",
    )

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Searching Endee and generating answer with Mistral…"):
                try:
                    answer, contexts = pipeline.ask(query)

                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("📄 Retrieved context chunks (from Endee)"):
                        if contexts:
                            for i, ctx in enumerate(contexts, 1):
                                st.markdown(
                                    f'<div class="context-item"><strong>Chunk {i}:</strong> {ctx}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.write("No context retrieved.")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Could not connect to Ollama. Make sure `ollama serve` is running "
                        f"at `{config.OLLAMA_BASE_URL}`."
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
